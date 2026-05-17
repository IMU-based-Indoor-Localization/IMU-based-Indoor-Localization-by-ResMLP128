package com.imulocal

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.os.Looper
import android.util.Log
import com.google.android.gms.location.FusedLocationProviderClient
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.LocationServices
import com.google.android.gms.location.Priority

/**
 * GpsTracker.kt
 * =============
 * FusedLocationProviderClient 를 사용해 GPS 위치를 수신하고 콜백으로 전달한다.
 *
 * 사용법:
 *   val gps = GpsTracker(context)
 *   gps.start { location -> /* lat/lng 처리 */ }
 *   gps.stop()
 *
 * 주의:
 *   - 호출 전 ACCESS_FINE_LOCATION 권한이 부여되어 있어야 한다.
 *   - start() 는 권한 확인 없이 직접 등록하므로 권한 체크는 호출자(MainActivity) 책임.
 */
class GpsTracker(context: Context) {

    companion object {
        private const val TAG = "GpsTracker"

        /** 위치 업데이트 최소 간격 (ms) */
        private const val INTERVAL_MS = 1_000L

        /** 위치 업데이트 최소 거리 — 0 = 거리 제한 없음 */
        private const val MIN_DIST_M = 0f
    }

    private val fusedClient: FusedLocationProviderClient =
        LocationServices.getFusedLocationProviderClient(context)

    private var callback: LocationCallback? = null

    // ── 시작 ──────────────────────────────────────────────────────
    /**
     * GPS 수신을 시작한다.
     * @param onLocation 위치 업데이트마다 메인 스레드에서 호출되는 콜백
     */
    @SuppressLint("MissingPermission")   // 권한 체크는 호출자 책임
    fun start(onLocation: (Location) -> Unit) {
        if (callback != null) {
            Log.w(TAG, "이미 실행 중 — 중복 start() 무시")
            return
        }

        val request = LocationRequest.Builder(
            Priority.PRIORITY_HIGH_ACCURACY,
            INTERVAL_MS
        )
            .setMinUpdateDistanceMeters(1.5f)   // 1.5m 미만 이동 시 콜백 무시 → 실내 GPS 노이즈 억제
            .build()

        callback = object : LocationCallback() {
            override fun onLocationResult(result: LocationResult) {
                val loc = result.lastLocation ?: return
                Log.v(TAG, "GPS: lat=${loc.latitude} lng=${loc.longitude} acc=${loc.accuracy}m")
                onLocation(loc)
            }
        }

        fusedClient.requestLocationUpdates(
            request,
            callback!!,
            Looper.getMainLooper()
        )
        Log.i(TAG, "GPS 수신 시작 (interval=${INTERVAL_MS}ms, PRIORITY_HIGH_ACCURACY)")
    }

    // ── 정지 ──────────────────────────────────────────────────────
    fun stop() {
        callback?.let {
            fusedClient.removeLocationUpdates(it)
            Log.i(TAG, "GPS 수신 정지")
        }
        callback = null
    }

    /** 현재 수신 중인지 여부 */
    fun isRunning(): Boolean = callback != null
}
