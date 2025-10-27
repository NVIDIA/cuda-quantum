if(CUDAQ_ENABLE_QIBO_BACKEND)
  target_sources(cudaq-rest-qpu PRIVATE QiboServerHelper.cpp)
  add_target_config(qibo)

  add_library(cudaq-serverhelper-qibo SHARED QiboServerHelper.cpp)
  target_link_libraries(cudaq-serverhelper-qibo
    PUBLIC
    cudaq-common
    fmt::fmt-header-only
  )
  install(TARGETS cudaq-serverhelper-qibo DESTINATION lib)
endif()
