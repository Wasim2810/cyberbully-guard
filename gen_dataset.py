"""
Synthetic dataset generator for the Cyberbullying Detection project.
Produces a realistic, labeled dataset (dataset/cyberbullying.csv) covering
6 classes: safe, offensive, toxic, cyberbullying, hate_speech, threat.

NOTE: This generator creates a representative sample dataset for
demonstration/training purposes since downloading the full Jigsaw /
Kaggle Cyberbullying dataset requires internet access. Swap this file
for the real Kaggle CSV (same column names: text,label) for production use.
"""
import random
import csv

random.seed(42)

# ---- Building blocks for each category ----
safe_templates = [
    "Have a wonderful day, {name}!",
    "Thanks so much for helping me with the project, {name}.",
    "I really enjoyed the movie last night.",
    "Can we meet tomorrow to discuss the report?",
    "Congratulations on your new job, {name}!",
    "The weather is beautiful today, perfect for a walk.",
    "I appreciate your feedback on my presentation.",
    "Let's grab coffee sometime this week.",
    "Great job on the exam, {name}, you worked hard for it.",
    "I love spending time with my family on weekends.",
    "Thank you for being such a supportive friend.",
    "The concert last night was amazing, wasn't it?",
    "I'm proud of how far you've come, {name}.",
    "Looking forward to our trip next month!",
    "Your cooking is absolutely delicious, {name}.",
    "Happy birthday {name}, hope you have a great year ahead.",
    "The team did a fantastic job on the launch.",
    "I hope you feel better soon, take care.",
    "Nice to meet you, welcome to the group.",
    "That was a really insightful comment, thanks for sharing.",
]

offensive_templates = [
    "Shut up, nobody asked for your opinion, {name}.",
    "You're so annoying, just stop talking.",
    "That was a stupid thing to say honestly.",
    "Get lost, no one wants you here.",
    "You are so dumb, how did you not get that?",
    "Ugh, you're so pathetic sometimes.",
    "Quit whining, it's so irritating.",
    "You're an idiot for thinking that would work.",
    "Nobody cares what you think, {name}.",
    "That's the dumbest idea I've ever heard.",
    "You always mess everything up, so annoying.",
    "Stop being so lazy and useless.",
    "You're such a loser, honestly.",
    "Why are you always so stupid about this?",
    "Just go away, you're so annoying.",
]

toxic_templates = [
    "You are a worthless piece of garbage, {name}.",
    "Everyone thinks you're disgusting and nobody wants you around.",
    "You're a pathetic excuse for a human being.",
    "I hope you fail at everything you do, loser.",
    "You're trash and everyone secretly hates you.",
    "Nobody will ever love someone as ugly and stupid as you.",
    "You disgust me, you worthless idiot.",
    "You're a complete failure and everyone knows it.",
    "Get out of here, you filthy waste of space.",
    "You are the most useless person I've ever met.",
    "I can't believe how repulsive and dumb you are.",
    "Everyone laughs at you behind your back, freak.",
    "You're an embarrassment to everyone around you.",
    "Why do you even bother, you're such garbage.",
]

cyberbullying_templates = [
    "You are useless and everyone hates you, {name}.",
    "Nobody likes you, just disappear from school forever.",
    "You should stop showing your ugly face online.",
    "Everyone in class is laughing at your pathetic posts.",
    "You're such a freak, no wonder you have no friends.",
    "Delete your account, nobody wants to see you here.",
    "You're a joke, everyone is mocking you behind your back.",
    "Stop posting, you're embarrassing yourself as always.",
    "You don't deserve any friends, you're that awful.",
    "We all agree you're the ugliest person in this group.",
    "Everyone in the group chat thinks you're pathetic and worthless.",
    "You're a loser online just like you are in real life.",
    "Nobody wants you in this server, just leave already.",
    "Your posts are cringe, everyone is laughing at you.",
    "You should be ashamed to even show your face here.",
]

hate_speech_templates = [
    "People like you don't belong in this country, {name}.",
    "I hate your entire kind, you're all the same.",
    "Your religion is disgusting and should be banned.",
    "People from your background are all criminals.",
    "Your race is the reason this country is failing.",
    "I despise everyone who looks or thinks like you.",
    "Your kind should never be allowed here again.",
    "Everyone from your community is worthless trash.",
    "I can't stand people of your background, all inferior.",
    "Your entire group is a disgrace to society.",
    "People like you should not have any rights here.",
    "I hate everything about your culture and beliefs.",
]

threat_templates = [
    "I will find you and hurt you if you post again, {name}.",
    "Watch your back, you're going to regret this.",
    "I know where you live, you better be careful.",
    "I'm going to make your life a living hell.",
    "You better stop or I will come after you.",
    "I'll make sure you pay for this, just wait.",
    "I am going to hurt you the next time I see you.",
    "You won't be safe anywhere after what you did.",
    "I will destroy everything you care about.",
    "Keep talking and see what happens to you next.",
    "I'll hunt you down if you don't stop right now.",
    "You will regret ever crossing me, I promise.",
]

names = ["Alex", "Sam", "Jordan", "Taylor", "Priya", "Rahul", "Emma", "Liam",
         "Sofia", "Noah", "Aisha", "Diego", "Mia", "Ethan", "Zara", "Kunal", ""]

def fill(t):
    name = random.choice(names)
    return t.format(name=name).replace("  ", " ").strip()

categories = {
    "safe": safe_templates,
    "offensive": offensive_templates,
    "toxic": toxic_templates,
    "cyberbullying": cyberbullying_templates,
    "hate_speech": hate_speech_templates,
    "threat": threat_templates,
}

rows = []
# Generate multiple variations per template using light augmentation
noise_suffixes = ["", " lol", "!!!", "...", " ??", " smh", " for real", " tbh", " literally"]
for label, templates in categories.items():
    for t in templates:
        for _ in range(18):  # augment each template into ~18 samples
            text = fill(t)
            suffix = random.choice(noise_suffixes)
            text = f"{text}{suffix}"
            rows.append((text, label))

random.shuffle(rows)

with open("dataset/cyberbullying.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["text", "label"])
    for text, label in rows:
        writer.writerow([text, label])

print(f"Generated {len(rows)} rows across {len(categories)} classes.")
