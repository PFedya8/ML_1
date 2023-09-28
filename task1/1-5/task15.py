from typing import List


def hello(name: str = None) -> str:
    if name is None or name == "":
        return "Hello!"
    else:
        return f"Hello, {name}!"


def int_to_roman(num: int) -> str:
    roman = {
        1: 'I',
        4: 'IV',
        5: 'V',
        9: 'IX',
        10: 'X',
        40: 'XL',
        50: 'L',
        90: 'XC',
        100: 'C',
        400: 'CD',
        500: 'D',
        900: 'CM',
        1000: 'M'
    }
    result = ""
    for val, numb in sorted(roman.items(), reverse=True):
        while num >= val:
            result += numb
            num -= val
    return result


def longest_common_prefix(strs_input: List[str]) -> str:
    if not strs_input:
        return ""
    strs_input = [s.lstrip() for s in strs_input]
    strs_input.sort()

    first = strs_input[0]
    last = strs_input[-1]
    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1
    return first[:i]


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5)+1):
        if n % i == 0:
            return False
    return True

def primes() -> int:
    n = 2
    while True:
        if is_prime(n):
            yield n
        n += 1



class BankCard:
    def __init__(self, total_sum: int, balance_limit: int = None):
        self.total_sum = total_sum
        self.balance_limit = balance_limit

    def __call__(self, sum_spent):
        if sum_spent > self.total_sum:
            raise ValueError("Not enough money to spend {} dollars.".format(sum_spent))
        self.total_sum -= sum_spent
        print(f"You spent {sum_spent} dollars.")

    def __str__(self):
        return "To learn the balance call balance."

    def __add__(self, other):
        new_card = BankCard(self.total_sum + other.total_sum)

        if self.balance_limit is not None and other.balance_limit is not None:
            new_card.balance_limit = max(self.balance_limit, other.balance_limit)
        elif self.balance_limit is None and other.balance_limit is None:
            new_card.balance_limit = None
        else:
            new_card.balance_limit = self.balance_limit if self.balance_limit is not None else other.balance_limit

        return new_card

    @property
    def balance(self):
        if self.balance_limit is not None:
            if self.balance_limit == 0:
                raise ValueError("Balance check limits exceeded.")
            self.balance_limit -= 1
        return self.total_sum

    def put(self, sum_put):
        self.total_sum += sum_put
        print(f"You put {sum_put} dollars.")


