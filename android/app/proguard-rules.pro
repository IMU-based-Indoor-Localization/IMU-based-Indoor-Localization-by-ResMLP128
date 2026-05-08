# =============================================================================
# proguard-rules.pro  —  IMU Indoor Localization (ResMLP128 + SC-EKF)
# =============================================================================
# R8/ProGuard 최소화 규칙.
# Play Store AAB 제출 시 네이티브 JNI, PyTorch Mobile Lite,
# Android Sensor API가 올바르게 동작하도록 보호합니다.
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# 1. JNI 네이티브 메서드 보호
#    EkfBridge.kt 의 external 선언과 EkfJniBridge.cpp 의 Java_com_imulocal_*
#    함수 이름이 일치해야 하므로 절대 난독화하지 않음.
# ─────────────────────────────────────────────────────────────────────────────
-keepclasseswithmembernames class * {
    native <methods>;
}

# EkfBridge 전체 보호 (object singleton + external 함수 이름)
-keep class com.imulocal.EkfBridge {
    *;
}


# ─────────────────────────────────────────────────────────────────────────────
# 2. PyTorch Mobile Lite (org.pytorch:pytorch_android_lite:2.1.0)
#    내부 JNI 브리지 및 리플렉션 진입점 보호
# ─────────────────────────────────────────────────────────────────────────────
-keep class org.pytorch.** { *; }
-keep class com.facebook.jni.** { *; }
-keep class com.facebook.soloader.** { *; }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Android Sensor / Location API
#    SensorEvent, SensorEventListener 구현체는 리플렉션으로 호출될 수 있음
# ─────────────────────────────────────────────────────────────────────────────
-keep class * implements android.hardware.SensorEventListener { *; }


# ─────────────────────────────────────────────────────────────────────────────
# 4. ViewModel / LiveData / Coroutines
#    androidx.lifecycle 는 리플렉션으로 생성자를 호출함
# ─────────────────────────────────────────────────────────────────────────────
-keep class * extends androidx.lifecycle.ViewModel { *; }
-keepclassmembers class * extends androidx.lifecycle.ViewModel {
    <init>(...);
}
# Coroutine 내부 Continuation 관련 (스택 트레이스 가독성 유지)
-keepnames class kotlinx.coroutines.internal.MainDispatcherFactory {}
-keepnames class kotlinx.coroutines.CoroutineExceptionHandler {}


# ─────────────────────────────────────────────────────────────────────────────
# 5. Kotlin 직렬화 / data class
#    JSON, Bundle 등에 쓰이는 data class 필드명 유지
# ─────────────────────────────────────────────────────────────────────────────
-keepclassmembers class com.imulocal.** {
    public <init>(...);
    public <fields>;
}


# ─────────────────────────────────────────────────────────────────────────────
# 6. Assets (모델 파일, 메타데이터 JSON)
#    assets/imu_model.ptl, assets/model_meta.json 은 코드가 아니므로
#    ProGuard 대상 아님 — 삭제되지 않도록 shrinkResources 예외 처리
# ─────────────────────────────────────────────────────────────────────────────
# (assets 폴더는 R8 shrinkResources 영향을 받지 않으므로 별도 규칙 불필요)


# ─────────────────────────────────────────────────────────────────────────────
# 7. 디버그 정보 (크래시 리포트 스택 트레이스 가독성)
# ─────────────────────────────────────────────────────────────────────────────
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile


# ─────────────────────────────────────────────────────────────────────────────
# 8. 불필요한 경고 억제
# ─────────────────────────────────────────────────────────────────────────────
-dontwarn org.pytorch.**
-dontwarn com.facebook.**
-dontwarn kotlin.Unit
