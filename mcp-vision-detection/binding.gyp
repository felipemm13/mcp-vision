{
  "targets": [
    {
      "target_name": "main",
      "sources": [
        "src/lib/Calibration_fixed.cpp",
        "src/lib/ConvertImage.cpp",
        "src/lib/ExtendedContour.cpp",
        "src/lib/FeetTracker.cpp",
        "src/lib/ComputerVisionWeb.cpp",
        "src/lib/MainAddon.cc",
        "src/main.cc"
      ],
      "include_dirs": [
        "<!(node -e \"require('nan')\")",
      ],
      'conditions': [
        ["OS=='linux'", {
          "cflags": ["-frtti", "-fexceptions"],
          "cflags_cc": ["-frtti", "-fexceptions"],
          "libraries": [
            "-lopencv_core",
            "-lopencv_imgproc",
            "-lopencv_highgui",
            "-lopencv_imgcodecs",
            '-lopencv_calib3d',
            "-lopencv_video",
            '-ljsoncpp'
          ]
        }]
      ]
    }
  ]
}
