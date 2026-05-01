//
//  MLXQwenAPI.swift
//  leanring-buddy
//
//  OpenAI-compatible chat client that talks to a local mlx_lm server.
//  Run the server with:
//    mlx_lm.server --model mlx-community/Qwen2.5-3B-Instruct-4bit --port 8080
//

import Foundation

/// Chat client for a locally-running mlx_lm server (OpenAI-compatible /v1/chat/completions).
/// Supports streaming SSE responses so text renders progressively on the cursor overlay.
class MLXQwenAPI {
    /// Base URL of the mlx_lm server. Change port if you started the server on a different one.
    static let defaultServerURL = "http://localhost:8080/v1/chat/completions"

    var model: String
    private let serverURL: URL
    private let session: URLSession

    init(serverURL: String = MLXQwenAPI.defaultServerURL, model: String = "mlx-community/Qwen2.5-3B-Instruct-4bit") {
        self.serverURL = URL(string: serverURL)!
        self.model = model

        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 120
        config.timeoutIntervalForResource = 300
        config.waitsForConnectivity = false  // localhost — no connectivity wait needed
        config.urlCache = nil
        config.httpCookieStorage = nil
        self.session = URLSession(configuration: config)
    }

    /// Sends a vision+text request to the local Qwen model with streaming.
    /// Images are base64-encoded and sent as OpenAI-style image_url content blocks.
    /// Calls `onTextChunk` on the main actor each time new text arrives.
    func analyzeImageStreaming(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)] = [],
        userPrompt: String,
        onTextChunk: @MainActor @Sendable (String) -> Void
    ) async throws -> (text: String, duration: TimeInterval) {
        let startTime = Date()

        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var messages: [[String: Any]] = []

        messages.append([
            "role": "system",
            "content": systemPrompt
        ])

        for (userPlaceholder, assistantResponse) in conversationHistory {
            messages.append(["role": "user", "content": userPlaceholder])
            messages.append(["role": "assistant", "content": assistantResponse])
        }

        // Build content blocks: label text + base64 image for each screenshot
        var contentBlocks: [[String: Any]] = []
        for image in images {
            contentBlocks.append([
                "type": "text",
                "text": image.label
            ])
            contentBlocks.append([
                "type": "image_url",
                "image_url": [
                    "url": "data:image/jpeg;base64,\(image.data.base64EncodedString())"
                ]
            ])
        }
        contentBlocks.append([
            "type": "text",
            "text": userPrompt
        ])
        messages.append(["role": "user", "content": contentBlocks])

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 1024,
            "stream": true,
            "messages": messages
        ]

        let bodyData = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = bodyData
        let payloadMB = Double(bodyData.count) / 1_048_576.0
        print("🤖 MLX Qwen streaming request: \(String(format: "%.1f", payloadMB))MB, \(images.count) image(s)")

        let (byteStream, response) = try await session.bytes(for: request)

        guard let httpResponse = response as? HTTPURLResponse else {
            throw NSError(
                domain: "MLXQwenAPI",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP response from mlx_lm server"]
            )
        }

        guard (200...299).contains(httpResponse.statusCode) else {
            var errorLines: [String] = []
            for try await line in byteStream.lines {
                errorLines.append(line)
            }
            throw NSError(
                domain: "MLXQwenAPI",
                code: httpResponse.statusCode,
                userInfo: [NSLocalizedDescriptionKey: "mlx_lm error (\(httpResponse.statusCode)): \(errorLines.joined(separator: "\n"))"]
            )
        }

        // Parse OpenAI-compatible SSE stream
        var accumulatedResponseText = ""

        for try await line in byteStream.lines {
            guard line.hasPrefix("data: ") else { continue }
            let jsonString = String(line.dropFirst(6))

            guard jsonString != "[DONE]" else { break }

            guard let jsonData = jsonString.data(using: .utf8),
                  let eventPayload = try? JSONSerialization.jsonObject(with: jsonData) as? [String: Any],
                  let choices = eventPayload["choices"] as? [[String: Any]],
                  let firstChoice = choices.first,
                  let delta = firstChoice["delta"] as? [String: Any],
                  let textChunk = delta["content"] as? String else {
                continue
            }

            accumulatedResponseText += textChunk
            let currentAccumulatedText = accumulatedResponseText
            await onTextChunk(currentAccumulatedText)
        }

        let duration = Date().timeIntervalSince(startTime)
        return (text: accumulatedResponseText, duration: duration)
    }

    /// Non-streaming fallback. Used for validation/element-detection requests
    /// where progressive rendering isn't needed.
    func analyzeImage(
        images: [(data: Data, label: String)],
        systemPrompt: String,
        conversationHistory: [(userPlaceholder: String, assistantResponse: String)] = [],
        userPrompt: String
    ) async throws -> (text: String, duration: TimeInterval) {
        let startTime = Date()

        var request = URLRequest(url: serverURL)
        request.httpMethod = "POST"
        request.timeoutInterval = 120
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        var messages: [[String: Any]] = []

        messages.append([
            "role": "system",
            "content": systemPrompt
        ])

        for (userPlaceholder, assistantResponse) in conversationHistory {
            messages.append(["role": "user", "content": userPlaceholder])
            messages.append(["role": "assistant", "content": assistantResponse])
        }

        var contentBlocks: [[String: Any]] = []
        for image in images {
            contentBlocks.append([
                "type": "text",
                "text": image.label
            ])
            contentBlocks.append([
                "type": "image_url",
                "image_url": [
                    "url": "data:image/jpeg;base64,\(image.data.base64EncodedString())"
                ]
            ])
        }
        contentBlocks.append([
            "type": "text",
            "text": userPrompt
        ])
        messages.append(["role": "user", "content": contentBlocks])

        let body: [String: Any] = [
            "model": model,
            "max_tokens": 256,
            "messages": messages
        ]

        let bodyData = try JSONSerialization.data(withJSONObject: body)
        request.httpBody = bodyData

        let (data, response) = try await session.data(for: request)

        guard let httpResponse = response as? HTTPURLResponse,
              (200...299).contains(httpResponse.statusCode) else {
            let responseString = String(data: data, encoding: .utf8) ?? "Unknown error"
            throw NSError(
                domain: "MLXQwenAPI",
                code: (response as? HTTPURLResponse)?.statusCode ?? -1,
                userInfo: [NSLocalizedDescriptionKey: "mlx_lm error: \(responseString)"]
            )
        }

        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard let choices = json?["choices"] as? [[String: Any]],
              let firstChoice = choices.first,
              let message = firstChoice["message"] as? [String: Any],
              let text = message["content"] as? String else {
            throw NSError(
                domain: "MLXQwenAPI",
                code: -1,
                userInfo: [NSLocalizedDescriptionKey: "Invalid response format from mlx_lm server"]
            )
        }

        let duration = Date().timeIntervalSince(startTime)
        return (text: text, duration: duration)
    }
}
