from .codes import Codes


class Formatter:
    @staticmethod
    def bold(message: str) -> str:
        return Codes.bold + message + Codes.end

    @staticmethod
    def green(message: str) -> str:
        return Codes.secondary_green + message + Codes.end

    @staticmethod
    def blue(message: str) -> str:
        return Codes.secondary_blue + message + Codes.end

    @staticmethod
    def yellow(message: str) -> str:
        return Codes.yellow + message + Codes.end

    @staticmethod
    def red(message: str) -> str:
        return Codes.red + message + Codes.end
