import readline


def timeout_input(prompt, timeout=None):
    try:
        user_input = input(prompt)
        return user_input
    except KeyboardInterrupt:
        return ""
