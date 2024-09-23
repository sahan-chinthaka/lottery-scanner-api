from functools import reduce
import re


def correct_number(data: list[str], number_count: int, number_length=1):
    full = reduce(lambda x, y: x + y, data, "").replace(" ", "", -1)

    full = re.sub(r"\D", "", full)

    numbers = []
    i = 0

    while i < len(full) and i // number_length < number_count:
        numbers.append(full[i : i + number_length])
        i += number_length

    return numbers


def correct_letter_number(data: list[str], number_count: int, number_length=2):
    full = reduce(lambda x, y: x + y, data, "").replace(" ", "", -1)
    letter = None

    if full[0].isalpha():
        letter = full[0]
        full = full[1:]

    full = re.sub(r"\D", "", full)

    numbers = ["" if letter is None else letter]
    i = 0

    while i < len(full) and i // number_length < number_count:
        numbers.append(full[i : i + number_length])
        i += number_length

    return numbers


def correct_date(data: list[str]):
    full = reduce(lambda x, y: x + y, data, "").replace(" ", "", -1)
    full = re.sub(r"[\D]", "", full)[:8]
    return f"{full[:4]}/{full[4:6]}/{full[6:8]}"


def correct_letter(data: list[str]):
    full = reduce(lambda x, y: x + y, data, "").replace(" ", "", -1)
    return full[0] if len(full) > 0 else ""
