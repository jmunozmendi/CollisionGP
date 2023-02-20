import dataclasses


@dataclasses.dataclass
class Codes:
    end = "\x1b[0m"
    bold = "\x1b[1m"
    dim = "\x1b[2m"
    italic = "\x1b[2m"
    underscore = "\x1b[4m"
    blink = "\x1b[5m"
    highlight = "\x1b[7m"
    hidden = "\x1b[8m"
    strikethrough = "\x1b[9m"
    double_underscore = "\x1b[21m"

    black = "\x1b[30m"
    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    white = "\x1b[37m"

    black__bg = "\x1b[40m"
    red_bg = "\x1b[41m"
    green_bg = "\x1b[42m"
    yellow_bg = "\x1b[43m"
    blue_bg = "\x1b[44m"
    magenta_bg = "\x1b[45m"
    cyan_bg = "\x1b[46m"
    white_bg = "\x1b[47m"

    secondary_black = "\x1b[90m"
    secondary_red = "\x1b[91m"
    secondary_green = "\x1b[92m"
    secondary_yellow = "\x1b[93m"
    secondary_blue = "\x1b[94m"
    secondary_magenta = "\x1b[95m"
    secondary_cyan = "\x1b[96m"
    secondary_white = "\x1b[97m"

    secondary_black_bg = "\x1b[40m"
    secondary_red_bg = "\x1b[41m"
    secondary_green_bg = "\x1b[42m"
    secondary_yellow_bg = "\x1b[43m"
    secondary_blue_bg = "\x1b[44m"
    secondary_magenta_bg = "\x1b[45m"
    secondary_cyan_bg = "\x1b[46m"
    secondary_white_bg = "\x1b[47m"

    line_up = "\x1b[1A"
    line_clear = "\x1b[2K"
