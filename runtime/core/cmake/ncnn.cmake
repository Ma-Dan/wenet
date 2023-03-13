if(NCNN)
  set(NCNN_URL "https://dan-1256867781.cos.ap-shanghai.myqcloud.com/ncnn-20221128-full-source.zip")
  set(URL_HASH "SHA256=4ee0ec08bea4952e9a4a2d526894d8c95881f58bec4e8c205cf453f3488ce737")

  set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(ncnn
    URL ${NCNN_URL}
    URL_HASH ${URL_HASH}
  )
  FetchContent_MakeAvailable(ncnn)

  include_directories(${ncnn_SOURCE_DIR}/src ${ncnn_BINARY_DIR}/src)
  link_directories(${ncnn_BINARY_DIR}/src)

  add_definitions(-DUSE_NCNN)
endif()
