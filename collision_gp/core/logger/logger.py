from .formatter import Formatter
from .codes import Codes


def output(function):

    def wrapper(*args, **kwargs):
        return_value = function(*args, **kwargs)

        print(return_value)
        if kwargs.get('flush'):
            print(Codes.line_up, end=Codes.line_clear)

    return wrapper


class Logger:

    @output
    @staticmethod
    def info(*message, sep: str = '', flush: bool = False) -> str:
        prefix: str = Formatter.green(Formatter.bold('[INFO]'))
        output_message: str = Formatter.green(sep.join(map(str, message)))
        return f'{prefix} {output_message}'

    @output
    @staticmethod
    def debug(*message, sep: str = '', flush: bool = False) -> str:
        prefix: str = Formatter.blue(Formatter.bold('[DEBUG]'))
        output_message: str = Formatter.blue(sep.join(map(str, message)))
        return f'{prefix} {output_message}'

    @output
    @staticmethod
    def warning(*message, sep: str = '', flush: bool = False) -> str:
        prefix: str = Formatter.yellow(Formatter.bold('[WARNING]'))
        output_message: str = Formatter.yellow(sep.join(map(str, message)))
        return f'{prefix} {output_message}'

    @output
    @staticmethod
    def error(*message, sep: str = '', flush: bool = False) -> str:
        prefix: str = Formatter.red(Formatter.bold('[ERROR]'))
        output_message: str = Formatter.red(sep.join(map(str, message)))
        return f'{prefix} {output_message}'
