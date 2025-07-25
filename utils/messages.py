GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

TICK = "[✓]"
CROSS = "[✗]"
WARN = "[!]"


def success(msg: str) -> str:
    return f"{GREEN}{TICK} {msg}{RESET}"


def fail(msg: str) -> str:
    return f"{RED}{CROSS} {msg}{RESET}"


def caution(msg: str) -> str:
    return f"{YELLOW}{WARN} {msg}{RESET}"
