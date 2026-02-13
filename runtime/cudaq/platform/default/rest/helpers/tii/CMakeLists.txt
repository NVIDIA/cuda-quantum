if(CUDAQ_ENABLE_TII_BACKEND)
  target_sources(cudaq-rest-qpu PRIVATE TiiServerHelper.cpp)
  add_target_config(tii)

  add_library(cudaq-serverhelper-tii SHARED TiiServerHelper.cpp)
  target_link_libraries(cudaq-serverhelper-tii
    PUBLIC
    cudaq-common
    fmt::fmt-header-only
  )
  install(TARGETS cudaq-serverhelper-tii DESTINATION lib)
endif()
