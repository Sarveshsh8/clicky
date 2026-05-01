//
//  ElevenLabsTTSClient.swift
//  leanring-buddy
//
//  Local TTS using AVSpeechSynthesizer — no network, no API keys.
//  Keeps the same interface as the ElevenLabs version so CompanionManager
//  doesn't need to change.
//
//  speakText() starts speaking and returns immediately (mirroring the
//  original AVAudioPlayer.play() behaviour). CompanionManager polls
//  isPlaying to detect when playback finishes.
//

import AVFoundation
import Foundation

@MainActor
final class ElevenLabsTTSClient: NSObject {
    private let synthesizer = AVSpeechSynthesizer()
    private var _isPlaying = false

    override init() {
        super.init()
        synthesizer.delegate = self
    }

    // Keeps CompanionManager's ElevenLabsTTSClient(proxyURL:) call compiling.
    convenience init(proxyURL: String) {
        self.init()
    }

    /// Starts speaking `text` and returns immediately.
    /// Caller can poll `isPlaying` to detect when playback finishes.
    func speakText(_ text: String) async throws {
        stopPlayback()
        try Task.checkCancellation()

        let utterance = AVSpeechUtterance(string: text)
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate
        utterance.pitchMultiplier = 1.0
        utterance.volume = 1.0

        _isPlaying = true
        synthesizer.speak(utterance)
        print("🔊 Local TTS: speaking \(text.count) chars")
        // Returns immediately — delegate callbacks update _isPlaying
    }

    var isPlaying: Bool { _isPlaying }

    func stopPlayback() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        _isPlaying = false
    }
}

extension ElevenLabsTTSClient: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor in self._isPlaying = false }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor in self._isPlaying = false }
    }
}
