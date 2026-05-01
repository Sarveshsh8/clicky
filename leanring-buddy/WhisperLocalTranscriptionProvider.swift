//
//  WhisperLocalTranscriptionProvider.swift
//  leanring-buddy
//
//  Upload-based transcription provider backed by a local whisper.cpp HTTP server.
//
//  Start the server with (from whisper.cpp repo root):
//    ./build/bin/whisper-server --model models/ggml-medium.bin --port 8081
//
//  The server exposes POST /inference which accepts multipart/form-data with a
//  WAV file and returns JSON: { "text": "..." }
//

import AVFoundation
import Foundation

struct WhisperLocalTranscriptionProviderError: LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

final class WhisperLocalTranscriptionProvider: BuddyTranscriptionProvider {
    /// URL of the local whisper.cpp server inference endpoint.
    static let defaultServerURL = "http://localhost:8081/inference"

    private let serverURL: URL

    let displayName = "Whisper (local)"
    let requiresSpeechRecognitionPermission = false

    var isConfigured: Bool {
        // Always considered configured — no API key needed, just a running server.
        true
    }

    var unavailableExplanation: String? {
        nil
    }

    init(serverURL: String = WhisperLocalTranscriptionProvider.defaultServerURL) {
        self.serverURL = URL(string: serverURL)!
    }

    func startStreamingSession(
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) async throws -> any BuddyStreamingTranscriptionSession {
        return WhisperLocalTranscriptionSession(
            serverURL: serverURL,
            keyterms: keyterms,
            onTranscriptUpdate: onTranscriptUpdate,
            onFinalTranscriptReady: onFinalTranscriptReady,
            onError: onError
        )
    }
}

private final class WhisperLocalTranscriptionSession: BuddyStreamingTranscriptionSession {
    // Whisper transcribes on key-up (upload-based), so we give it 10s before
    // forcing a final transcript to avoid hanging if the server is slow.
    let finalTranscriptFallbackDelaySeconds: TimeInterval = 10.0

    private struct WhisperInferenceResponse: Decodable {
        let text: String
    }

    private static let targetSampleRate = 16_000

    private let serverURL: URL
    private let keyterms: [String]
    private let onTranscriptUpdate: (String) -> Void
    private let onFinalTranscriptReady: (String) -> Void
    private let onError: (Error) -> Void

    private let stateQueue = DispatchQueue(label: "com.learningbuddy.whisper.transcription")
    private let audioPCM16Converter = BuddyPCM16AudioConverter(
        targetSampleRate: Double(targetSampleRate)
    )
    private let urlSession: URLSession

    private var bufferedPCM16AudioData = Data()
    private var hasRequestedFinalTranscript = false
    private var hasDeliveredFinalTranscript = false
    private var isCancelled = false
    private var transcriptionUploadTask: Task<Void, Never>?

    init(
        serverURL: URL,
        keyterms: [String],
        onTranscriptUpdate: @escaping (String) -> Void,
        onFinalTranscriptReady: @escaping (String) -> Void,
        onError: @escaping (Error) -> Void
    ) {
        self.serverURL = serverURL
        self.keyterms = keyterms
        self.onTranscriptUpdate = onTranscriptUpdate
        self.onFinalTranscriptReady = onFinalTranscriptReady
        self.onError = onError

        let urlSessionConfiguration = URLSessionConfiguration.default
        urlSessionConfiguration.timeoutIntervalForRequest = 60
        urlSessionConfiguration.timeoutIntervalForResource = 120
        urlSessionConfiguration.waitsForConnectivity = false  // localhost
        self.urlSession = URLSession(configuration: urlSessionConfiguration)
    }

    func appendAudioBuffer(_ audioBuffer: AVAudioPCMBuffer) {
        guard let audioPCM16Data = audioPCM16Converter.convertToPCM16Data(from: audioBuffer),
              !audioPCM16Data.isEmpty else {
            return
        }

        stateQueue.async {
            guard !self.hasRequestedFinalTranscript, !self.isCancelled else { return }
            self.bufferedPCM16AudioData.append(audioPCM16Data)
        }
    }

    func requestFinalTranscript() {
        stateQueue.async {
            guard !self.hasRequestedFinalTranscript, !self.isCancelled else { return }
            self.hasRequestedFinalTranscript = true

            let bufferedPCM16AudioData = self.bufferedPCM16AudioData
            self.transcriptionUploadTask = Task { [weak self] in
                await self?.transcribeBufferedAudio(bufferedPCM16AudioData)
            }
        }
    }

    func cancel() {
        transcriptionUploadTask?.cancel()
        transcriptionUploadTask = nil
        // Capture session locally so invalidation doesn't retain self past deinit
        let sessionToInvalidate = urlSession
        stateQueue.async {
            self.isCancelled = true
            self.bufferedPCM16AudioData.removeAll(keepingCapacity: false)
            sessionToInvalidate.invalidateAndCancel()
        }
    }

    private func transcribeBufferedAudio(_ bufferedPCM16AudioData: Data) async {
        guard !Task.isCancelled else { return }

        let audioIsEmpty = stateQueue.sync {
            isCancelled || bufferedPCM16AudioData.isEmpty
        }

        if audioIsEmpty {
            deliverFinalTranscript("")
            return
        }

        let wavAudioData = BuddyWAVFileBuilder.buildWAVData(
            fromPCM16MonoAudio: bufferedPCM16AudioData,
            sampleRate: Self.targetSampleRate
        )

        do {
            let transcriptText = try await requestWhisperTranscription(for: wavAudioData)
            guard !stateQueue.sync(execute: { isCancelled }) else { return }

            if !transcriptText.isEmpty {
                onTranscriptUpdate(transcriptText)
            }

            deliverFinalTranscript(transcriptText)
        } catch {
            guard !stateQueue.sync(execute: { isCancelled }) else { return }
            print("[Whisper Transcription] ❌ Upload failed (\(wavAudioData.count) bytes): \(error.localizedDescription)")
            onError(error)
        }
    }

    private func requestWhisperTranscription(for wavAudioData: Data) async throws -> String {
        let multipartBoundary = "Boundary-\(UUID().uuidString)"
        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.setValue("multipart/form-data; boundary=\(multipartBoundary)", forHTTPHeaderField: "Content-Type")

        let requestBodyData = makeMultipartRequestBody(
            boundary: multipartBoundary,
            wavAudioData: wavAudioData
        )
        request.httpBody = requestBodyData

        let (responseData, response) = try await urlSession.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw WhisperLocalTranscriptionProviderError(
                message: "Whisper server returned an invalid response."
            )
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            let responseText = String(data: responseData, encoding: .utf8) ?? "Unknown error"
            throw WhisperLocalTranscriptionProviderError(
                message: "Whisper server failed (\(httpResponse.statusCode)): \(responseText)"
            )
        }

        // whisper.cpp server returns { "text": "..." }
        if let inferenceResponse = try? JSONDecoder().decode(
            WhisperInferenceResponse.self,
            from: responseData
        ) {
            return inferenceResponse.text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Plain-text fallback in case the server returns text directly
        let plainText = String(data: responseData, encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""

        if !plainText.isEmpty {
            return plainText
        }

        throw WhisperLocalTranscriptionProviderError(
            message: "Whisper server returned an empty transcript."
        )
    }

    private func makeMultipartRequestBody(boundary: String, wavAudioData: Data) -> Data {
        var requestBodyData = Data()

        // whisper.cpp server expects the audio file field named "file"
        requestBodyData.appendMultipartFileFieldForWhisper(
            named: "file",
            filename: "voice-input.wav",
            mimeType: "audio/wav",
            fileData: wavAudioData,
            usingBoundary: boundary
        )

        // Pass response_format=json so we get { "text": "..." } back
        requestBodyData.appendMultipartFormFieldForWhisper(
            named: "response_format",
            value: "json",
            usingBoundary: boundary
        )

        requestBodyData.appendStringForWhisper("--\(boundary)--\r\n")

        return requestBodyData
    }

    private func deliverFinalTranscript(_ transcriptText: String) {
        guard !hasDeliveredFinalTranscript else { return }
        hasDeliveredFinalTranscript = true
        onFinalTranscriptReady(transcriptText)
    }

    deinit {
        cancel()
    }
}

private extension Data {
    mutating func appendStringForWhisper(_ string: String) {
        append(string.data(using: .utf8)!)
    }

    mutating func appendMultipartFormFieldForWhisper(
        named fieldName: String,
        value: String,
        usingBoundary boundary: String
    ) {
        appendStringForWhisper("--\(boundary)\r\n")
        appendStringForWhisper("Content-Disposition: form-data; name=\"\(fieldName)\"\r\n\r\n")
        appendStringForWhisper("\(value)\r\n")
    }

    mutating func appendMultipartFileFieldForWhisper(
        named fieldName: String,
        filename: String,
        mimeType: String,
        fileData: Data,
        usingBoundary boundary: String
    ) {
        appendStringForWhisper("--\(boundary)\r\n")
        appendStringForWhisper("Content-Disposition: form-data; name=\"\(fieldName)\"; filename=\"\(filename)\"\r\n")
        appendStringForWhisper("Content-Type: \(mimeType)\r\n\r\n")
        append(fileData)
        appendStringForWhisper("\r\n")
    }
}
