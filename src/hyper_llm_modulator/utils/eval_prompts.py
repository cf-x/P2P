# 3-shot ICL
IN_CONTEXT_EXAMPLES = {
    "gsm8k": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May.
Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.
#### 72

## Example 2
Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?

Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

## Example 3
Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?

Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.
Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.
This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.
#### 5
""",
    "humaneval": "",  # usually 0-shot
    "mbpp": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Instruction: Write a function to find the shared elements from the given two lists. Your code should statisfy the following assertions:
```python
assert set(similar_elements((3, 4, 5, 6),(5, 7, 4, 10))) == set((4, 5))
assert set(similar_elements((1, 2, 3, 4),(5, 4, 3, 7))) == set((3, 4))
assert set(similar_elements((11, 12, 14, 13),(17, 15, 14, 13))) == set((13, 14))
```

```python
def similar_elements(test_tup1, test_tup2):
    res = tuple(set(test_tup1) & set(test_tup2))
    return (res) 
```

## Example 2
Instruction: Write a python function to identify non-prime numbers. Your code should statisfy the following assertions:
```python
assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True
assert is_not_prime(37) == False
```

```python
import math
def is_not_prime(n):
    result = False
    for i in range(2,int(math.sqrt(n)) + 1):
        if n % i == 0:
            result = True
    return result
```

## Example 3
Instruction: Write a function to find the n largest integers from a given list of numbers, returned in descending order. Your code should statisfy the following assertions:
```python
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],3)==[85, 75, 65]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],2)==[85, 75]
assert heap_queue_largest( [25, 35, 22, 85, 14, 65, 75, 22, 58],5)==[85, 75, 65, 58, 35]
```

```python
import heapq as hq
def heap_queue_largest(nums,n):
    largest_nums = hq.nlargest(n, nums)
    return largest_nums
```
""",
    "boolq": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Passage: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet.

Question: do iran and afghanistan speak the same language

Answer: True

## Example 2
Passage: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable.

Question: do good samaritan laws protect those who help at an accident

Answer: True

## Example 3
Passage: Windows Movie Maker (formerly known as Windows Live Movie Maker in Windows 7) is a discontinued video editing software by Microsoft. It is a part of Windows Essentials software suite and offers the ability to create and edit videos as well as to publish them on OneDrive, Facebook, Vimeo, YouTube, and Flickr.

Question: is windows movie maker part of windows essentials

Answer: True
""",
    "winogrande": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Sentence: John moved the couch from the garage to the backyard to create space. The _ is small.

Option 1: garage

Option 2: backyard

Answer: 1

## Example 2
Sentence: The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently.

Option 1: Justin

Option 2: Robert

Answer: 2

## Example 3
Sentence: Dennis drew up a business proposal to present to Logan because _ wants his investment.

Option 1: Dennis

Option 2: Logan

Answer: 1
""",
    "piqa": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: When boiling butter, when it's ready, you can

0: Pour it onto a plate

1: Pour it into a jar

Answer: 1

## Example 2
Question: To permanently attach metal legs to a chair, you can

0: Weld the metal together to get it to stay firmly in place

1: Nail the metal together to get it to stay firmly in place

Answer: 0

## Example 3
Question: how do you indent something?

0: leave a space before starting the writing

1: press the spacebar

Answer: 0

""",
    "hellaswag": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Passage: Then, the man writes over the snow covering the window of a car, and a woman wearing winter clothes smiles. then

0: , the man adds wax to the windshield and cuts it.

1: , a person board a ski lift, while two men supporting the head of the person wearing winter clothes snow as the we girls sled.

2: , the man puts on a christmas coat, knitted with netting.

3: , the man continues removing the snow on his car.

Answer: 3

## Example 2
Passage: A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. the pans

0: contain egg yolks and baking soda.

1: are then sprinkled with brown sugar.

2: are placed in a strainer on the counter.

3: are filled with pastries and loaded into the oven.

Answer: 3

## Example 3
Passage: A female chef in white uniform shows a stack of baking pans in a large kitchen presenting them. The pans are filled with pastries and loaded into the oven. a knife

0: is seen moving on a board and cutting out its contents.

1: hits the peeled cheesecake, followed by sliced custard and still cooked ice cream.

2: etches a shape into the inside of the baked pans.

3: is used to cut cylinder shaped dough into rounds.

Answer: 3
""",
    "arc_easy": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: Which factor will most likely cause a person to develop a fever?
A: a leg muscle relaxing after exercise
B: a bacterial population in the bloodstream
C: several viral particles on the skin
D: carbohydrates being digested in the stomach

Answer: B

## Example 2
Question: Lichens are symbiotic organisms made of green algae and fungi. What do the green algae supply to the fungi in this symbiotic relationship?
A: carbon dioxide
B: food
C: protection
D: water

Answer: B

## Example 3
Question: When a switch is used in an electrical circuit, the switch can
A: cause the charge to build.
B: increase and decrease the voltage.
C: cause the current to change direction.
D: stop and start the flow of current.

Answer: D
""",
    "arc_challenge": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: George wants to warm his hands quickly by rubbing them. Which skin surface will produce the most heat?
A: dry palms
B: wet palms
C: palms covered with oil
D: palms covered with lotion

Answer: A

## Example 2
Question: Which of the following statements best explains why magnets usually stick to a refrigerator door?
A: The refrigerator door is smooth.
B: The refrigerator door contains iron.
C: The refrigerator door is a good conductor.
D: The refrigerator door has electric wires in it.

Answer: B

## Example 3
Question: A fold observed in layers of sedimentary rock most likely resulted from the
A: cooling of flowing magma.
B: converging of crustal plates.
C: deposition of river sediments.
D: solution of carbonate minerals.

Answer: B
""",
    "openbookqa": """
Here are some examples of the tasks you will be asked to solve.

## Example 1
Question: The sun is responsible for
A: puppies learning new tricks
B: children growing up and getting old
C: flowers wilting in a vase
D: plants sprouting, blooming and wilting

Answer: D

## Example 2
Question: When standing miles away from Mount Rushmore
A: the mountains seem very close
B: the mountains are boring
C: the mountains look the same as from up close
D: the mountains seem smaller than in photographs

Answer: D

## Example 3
Question: When food is reduced in the stomach
A: the mind needs time to digest
B: take a second to digest what I said
C: nutrients are being deconstructed
D: reader's digest is a body of works

Answer: C
""",
}

TASK_DESCRIPTION_MESSAGES = {
    "gsm8k": "You will be tasked with interpreting mathematical situations described in words. The goal is to use logical reasoning and calculations to determine the numerical answers based on the context provided.",
    "humaneval": "Engage in building distinct functions that meet the requirements of various presented problems, honing your ability to translate problem statements into logical code. Utilize structured thinking to implement efficient solutions.",
    "mbpp": "Your challenge is to solve a series of problems by writing functions in Python. These problems require handling lists and strings, allowing you to showcase your proficiency in coding while addressing practical programming scenarios.",
    "boolq": "Analyze the given details about various subjects, including movies, sports, and television shows. Your role is to confirm whether certain claims are true or false.",
    "winogrande": "In this exercise, you need to read short narratives and discern which person or object fits best within the context of the sentence.",
    "piqa": "You will explore practical questions and select an answer that presents a logical and widely accepted approach to solve a given problem or complete a task successfully.",
    "hellaswag": "This task revolves around completing an unfinished text by selecting an ending that matches its tone and context. It requires you to think critically about how narratives develop and conclude effectively.",
    "arc_easy": "Your job is to discern which information best answers a posed question, focusing on practical examples and scientific principles. This requires a strong grasp of underlying concepts in ecology or physics.",
    "arc_challenge": "This task is about analyzing questions which examine your grasp of scientific ideas. You must connect conceptual knowledge with practical examples from geology, ecology and environmental changes.",
    "openbookqa": "Analyze the provided statements carefully and determine which one best fits into the context of the passage. This requires comprehension skills and the ability to make logical inferences.",
}
