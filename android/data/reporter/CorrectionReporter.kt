package com.sitecoach.data.reporter

import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL

/**
 * Submits user-reported STT mistakes back to the Site Coach backend so
 * the seed dictionary can grow organically over time.
 *
 * Wire it up from your UI by showing a "Wrong transcription? Report it"
 * button next to each transcribed line; on tap, ask the user for the
 * correct word, then call:
 *
 *   CorrectionReporter("https://your-backend.example.com").report(
 *       wrongWord = "पापा", correctWord = "वापस",
 *       audioId = "uuid", userId = "alok"
 *   )
 *
 * The matching backend endpoint is `POST /report-correction` (see main.py).
 */
class CorrectionReporter(
    private val backendBaseUrl: String,
    private val timeoutMs: Int = 8_000,
) {

    companion object {
        private const val TAG = "CorrectionReporter"
        private const val ENDPOINT = "/report-correction"
    }

    sealed class Result {
        data class Success(val responseJson: String) : Result()
        data class Failure(val httpStatus: Int, val message: String) : Result()
    }

    suspend fun report(
        wrongWord: String,
        correctWord: String,
        audioId: String? = null,
        userId: String? = null,
        category: String = "USER_REPORT",
        notes: String = "",
    ): Result = withContext(Dispatchers.IO) {
        if (wrongWord.isBlank() || correctWord.isBlank()) {
            return@withContext Result.Failure(
                httpStatus = 0,
                message = "wrongWord and correctWord cannot be empty",
            )
        }

        val payload = JSONObject().apply {
            put("wrongWord", wrongWord.trim())
            put("correctWord", correctWord.trim())
            put("category", category)
            put("notes", notes)
            audioId?.let { put("audioId", it) }
            userId?.let { put("userId", it) }
        }

        val url = URL(backendBaseUrl.trimEnd('/') + ENDPOINT)
        val conn = (url.openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            connectTimeout = timeoutMs
            readTimeout = timeoutMs
            doInput = true
            doOutput = true
            setRequestProperty("Content-Type", "application/json; charset=UTF-8")
            setRequestProperty("Accept", "application/json")
        }

        try {
            conn.outputStream.use { it.write(payload.toString().toByteArray(Charsets.UTF_8)) }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val body = stream?.bufferedReader()?.use { it.readText() } ?: ""

            if (code in 200..299) {
                Log.d(TAG, "Reported $wrongWord -> $correctWord (HTTP $code)")
                Result.Success(body)
            } else {
                Log.w(TAG, "Report failed HTTP $code: $body")
                Result.Failure(code, body)
            }
        } catch (e: Exception) {
            Log.e(TAG, "Report network failure", e)
            Result.Failure(httpStatus = 0, message = e.message ?: "network error")
        } finally {
            conn.disconnect()
        }
    }
}
