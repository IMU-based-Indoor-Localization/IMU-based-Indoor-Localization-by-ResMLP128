cmake_minimum_required(VERSION 3.22.1)
project(imu_ekf_jni)

set(CMAKE_CXX_STANDARD 17)

# ── Eigen: FetchContent로 자동 다운로드 ───────────────────────
# 수동 설치 불필요. 첫 빌드 시 CMake가 자동으로 받아옴.
# (오프라인 환경이면 third_party/eigen 에 직접 압축 해제해도 됨)
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# third_party/eigen 에 이미 있으면 다운로드 생략
# 우선순위: ① cpp/third_party/eigen (CHECKLIST.md 지침 경로)
#           ② 프로젝트 루트/third_party/eigen (이전 경로, fallback)
set(EIGEN_LOCAL_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
if(NOT EXISTS "${EIGEN_LOCAL_DIR}/Eigen/Dense")
    set(EIGEN_LOCAL_DIR "${CMAKE_SOURCE_DIR}/../../../../../third_party/eigen")
endif()
if(EXISTS "${EIGEN_LOCAL_DIR}/Eigen/Dense")
    message(STATUS "Eigen: 로컬 캐시 사용 (${EIGEN_LOCAL_DIR})")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${EIGEN_LOCAL_DIR}")
else()
    message(STATUS "Eigen: FetchContent로 다운로드 중 (3.4.0) ...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    # FetchContent_MakeAvailable 대신 Populate 사용:
    # Eigen의 CMakeLists.txt를 실행하지 않고 헤더 경로만 추가.
    # → SuperLU 등 외부 의존성 탐색 오류를 완전히 차단함.
    FetchContent_GetProperties(eigen)
    if(NOT eigen_POPULATED)
        FetchContent_Populate(eigen)
    endif()
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    target_include_directories(Eigen3::Eigen INTERFACE "${eigen_SOURCE_DIR}")
endif()

# ── JNI 공유 라이브러리 ────────────────────────────────────────
add_library(
    imu_ekf_jni
    SHARED
    ekf/imu_ekf.cpp
    EkfJniBridge.cpp
)

target_include_directories(imu_ekf_jni PRIVATE ekf/)
target_link_libraries(imu_ekf_jni Eigen3::Eigen)

# Android 표준 시스템 라이브러리
find_library(log-lib log)
target_link_libraries(imu_ekf_jni ${log-lib})
