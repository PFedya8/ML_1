def check(x: str, file: str):
    words = x.split()
    word_count = {}

    for word in words:
        word = word.lower()
        word_count[word] = word_count.get(word, 0) + 1

    sorted_keys = sorted(word_count.keys())
    with open(file, "w") as f:
        for key in sorted_keys:
            f.write(f"{key} {word_count[key]}\n")


def main():
    check("a aa abC aa ac abc bcd a", "file.txt")


if __name__ == '__main__':
    main()