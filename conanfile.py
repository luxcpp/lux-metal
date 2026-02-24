from conan import ConanFile
from conan.tools.cmake import CMake, cmake_layout
from conan.errors import ConanInvalidConfiguration


class LuxMetal(ConanFile):
    name = "lux-metal"
    package_type = "shared-library"  # Plugin - always shared
    settings = "os", "arch", "compiler", "build_type"
    options = {"fPIC": [True, False]}
    default_options = {"fPIC": True}

    exports_sources = "CMakeLists.txt", "include/*", "src/*", "LICENSE*"

    # Only need lux-accel for the backend_api.h header at build time
    # No runtime dependency - plugin is loaded by lux-accel
    build_requires = "lux-accel/0.1.0"

    def validate(self):
        if self.settings.os != "Macos":
            raise ConanInvalidConfiguration("lux-metal requires macOS")

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
        # Plugin - no CMake target export needed
        # Consumers don't link against this, they load it at runtime
        self.cpp_info.libs = []  # No libs to link
        self.cpp_info.bindirs = []
        self.cpp_info.libdirs = ["lib/lux/plugins"]
