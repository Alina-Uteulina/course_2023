import json


def main(input):
    # Тут код
    return {"a":"b"}


if __name__ == "__main__":

    with open("C:/Users/rache/Desktop/rab/course_20233/homeworks/homework_1/input_data/1.json", "r") as file:
        data = json.load(file)

    output = main(data)

    with open(r"output.json", "w", ) as  file:
        json.dump(output, file)