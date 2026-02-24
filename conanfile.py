from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.errors import ConanInvalidConfiguration


class LuxMetal(ConanFile):
    name = "lux-metal"
    version = "0.1.0"
    license = "BSD-3-Clause"
    description = "Metal backend plugin for lux-accel (macOS / iOS)"
    package_type = "shared-library"  # Plugin - always shared
    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    exports_sources = (
        "CMakeLists.txt",
        "include/*",
        "src/*",
        "test/*",
        "LICENSE*",
    )

    # The plugin only needs backend_api.h from lux-accel at build time. Metal,
    # Foundation and QuartzCore frameworks are linked via CMake.
    def build_requirements(self):
        self.tool_requires("lux-accel/0.1.0")

    def validate(self):
        if self.settings.os not in ("Macos", "iOS"):
            raise ConanInvalidConfiguration(
                "lux-metal requires macOS or iOS (Metal framework)"
            )

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def layout(self):
        cmake_layout(self)

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

    def package(self):
        cmake = CMake(self)
        cmake.install()

    def package_info(self):
        # Plugin - no CMake target export needed; consumers dlopen it.
        self.cpp_info.libs = []
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = ["lib/lux/plugins"]
