from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


FORBIDDEN_TERMS = {
    "ad",
    "alzheimer",
    "amyloid",
    "biomedical",
    "brain",
    "cell",
    "clinical",
    "dementia",
    "disease",
    "dna",
    "gene",
    "genetic",
    "genome",
    "immune",
    "medical",
    "medicine",
    "microglia",
    "mutation",
    "neuron",
    "pathology",
    "patient",
    "protein",
    "rna",
    "synapse",
    "tau",
}


TEMPLATES: list[tuple[str, str]] = [
    ("city", "The {place} opened near the {location} on {day}."),
    ("city", "Local officials discussed a new {project} for the {district}."),
    ("city", "Residents gathered at the {venue} to discuss {topic}."),
    ("sports", "The {team} won the {event} after a close final round."),
    ("sports", "Fans waited outside the {venue} before the {event}."),
    ("sports", "The coach changed the {strategy} before the second half."),
    ("weather", "The forecast predicted {weather} across the {region} this weekend."),
    ("weather", "Strong {condition} delayed travel near the {location}."),
    ("weather", "Clear skies returned after {condition} passed through the area."),
    ("food", "The {place} introduced a menu with {food_item} and {drink}."),
    ("food", "The recipe used {ingredient_a}, {ingredient_b}, and {ingredient_c}."),
    ("food", "Customers ordered {food_item} during the lunch rush."),
    ("education", "Students completed a {assignment} before the end of the semester."),
    ("education", "The school added a course about {subject}."),
    ("education", "Teachers prepared a {activity} for the morning class."),
    ("travel", "Tourists visited the {landmark} during the {season} festival."),
    ("travel", "The hotel offered a discount for {duration} stays."),
    ("travel", "Passengers boarded the {transport} shortly before departure."),
    ("transportation", "The {transport} schedule changed during the {season} season."),
    ("transportation", "A new route connected the {district} with the {location}."),
    ("transportation", "Commuters reported delays near the {location} station."),
    ("home", "The family placed a {furniture} beside the {room} window."),
    ("home", "They repaired the {house_part} after the storm."),
    ("home", "New shelves were installed in the {room}."),
    ("entertainment", "The {event} opened with a documentary about {subject}."),
    ("entertainment", "A local band performed at the {venue} on {day} night."),
    ("entertainment", "The theater announced its schedule for the {season}."),
    ("business", "The company reported higher {metric} during the quarter."),
    ("business", "The board approved a budget for the {campaign}."),
    ("business", "Employees attended a workshop about {office_topic}."),
    ("technology", "The software update added new {feature} settings."),
    ("technology", "Users can sync {digital_item} across multiple devices."),
    ("technology", "The website redesigned its {interface_part} for mobile screens."),
    ("arts", "The artist painted {scene} using bright colors."),
    ("arts", "A sculpture was installed near the {location}."),
    ("arts", "Children made {craft_item} during the workshop."),
    ("nature", "The trail follows a path through the {landscape}."),
    ("nature", "Several boats crossed the {water_body} before sunset."),
    ("nature", "Autumn leaves covered the road near the {landmark}."),
    ("shopping", "The store started a sale on {product}."),
    ("shopping", "Customers compared prices before choosing a {product}."),
    ("shopping", "Online orders will arrive within {duration}."),
    ("culture", "The annual parade attracted visitors from nearby {place_plural}."),
    ("culture", "A traditional performance closed the evening {event}."),
    ("culture", "The festival featured {craft_item}, music, and regional food."),
    ("finance", "The bank introduced a new app for {finance_task}."),
    ("finance", "The committee reviewed the quarterly {document}."),
    ("finance", "The invoice must be paid before the end of the {period}."),
    ("work", "The manager scheduled a meeting for {day} afternoon."),
    ("work", "The team prepared a slide deck for the {meeting_type}."),
    ("work", "New employees received badges on their first day."),
    ("communication", "The newsletter included updates about upcoming {event}."),
    ("communication", "A short message reminded guests to arrive {time_phrase}."),
    ("communication", "The editor revised the headline before publication."),
    ("daily_life", "The alarm clock rang before {time_of_day}."),
    ("daily_life", "She carried a {object} on the way to the {location}."),
    ("daily_life", "The notebook was left on the {furniture}."),
    ("public_service", "The department held a safety demonstration at the {place}."),
    ("public_service", "A recycling program started in several {building_type}."),
    ("public_service", "Residents received instructions about {public_notice}."),
    ("books", "The novel follows a family moving to a {setting}."),
    ("books", "Readers discussed the ending during the {group} meeting."),
    ("books", "The publisher released a revised edition with a new cover."),
    ("events", "The registration desk opened at {time_phrase}."),
    ("events", "The reception was held in a garden behind the {place}."),
    ("events", "The ceremony ended with a group photograph."),
    ("clothing", "The jacket was made from {fabric} fabric."),
    ("clothing", "The tailor adjusted the {clothing_part} before the final fitting."),
    ("clothing", "The store displayed {clothing_item} near the front window."),
    ("tools", "The carpenter measured the {material} before cutting it."),
    ("tools", "The repair kit included a {tool}, tape, and small screws."),
    ("tools", "The toolbox was placed on a shelf near the door."),
]


VALUES: dict[str, list[str]] = {
    "activity": ["reading session", "map exercise", "drawing task", "group project", "writing lesson"],
    "assignment": ["history report", "design project", "reading journal", "math worksheet", "language exercise"],
    "building_type": ["apartment buildings", "office towers", "community halls", "school buildings", "shopping centers"],
    "campaign": ["marketing campaign", "customer survey", "product launch", "training program", "public event"],
    "clothing_item": ["summer dresses", "wool scarves", "linen shirts", "rain jackets", "canvas shoes"],
    "clothing_part": ["sleeves", "collar", "hem", "cuffs", "waistband"],
    "condition": ["winds", "rain", "fog", "snow", "clouds"],
    "craft_item": ["paper lanterns", "wooden frames", "clay bowls", "woven baskets", "painted tiles"],
    "day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    "digital_item": ["calendar events", "personal notes", "photo albums", "task lists", "bookmarks"],
    "district": ["museum district", "harbor district", "market district", "university district", "theater district"],
    "document": ["budget report", "planning document", "expense summary", "sales report", "proposal"],
    "drink": ["iced tea", "lemonade", "coffee", "sparkling water", "apple juice"],
    "duration": ["weekend", "three-day", "weekly", "short", "holiday"],
    "event": ["festival", "concert", "conference", "parade", "exhibition"],
    "fabric": ["lightweight waterproof", "soft cotton", "dark wool", "striped linen", "recycled canvas"],
    "feature": ["privacy", "camera", "display", "keyboard", "notification"],
    "finance_task": ["account management", "monthly budgeting", "invoice tracking", "card payments", "savings planning"],
    "food_item": ["grilled vegetables", "fresh noodles", "tomato soup", "rice bowls", "roasted potatoes"],
    "furniture": ["wooden desk", "blue chair", "small table", "lamp stand", "bookshelf"],
    "group": ["monthly book club", "student reading group", "library circle", "evening discussion group", "weekend club"],
    "house_part": ["garden fence", "front gate", "kitchen cabinet", "window frame", "garage door"],
    "ingredient_a": ["flour", "rice", "oats", "beans", "pasta"],
    "ingredient_b": ["sugar", "salt", "olive oil", "butter", "pepper"],
    "ingredient_c": ["eggs", "cheese", "tomatoes", "carrots", "mushrooms"],
    "interface_part": ["navigation menu", "search page", "login screen", "settings panel", "home page"],
    "landmark": ["old castle", "stone bridge", "clock tower", "public garden", "historic harbor"],
    "landscape": ["pine forest", "green valley", "rocky hillside", "quiet meadow", "coastal dunes"],
    "location": ["station", "park", "harbor", "library", "plaza", "airport"],
    "material": ["board", "wooden panel", "metal sheet", "tile", "plastic pipe"],
    "meeting_type": ["client presentation", "planning meeting", "annual review", "design review", "budget meeting"],
    "metric": ["sales", "orders", "attendance", "revenue", "membership"],
    "object": ["blue umbrella", "red notebook", "small backpack", "paper bag", "silver key"],
    "office_topic": ["customer service", "time management", "team planning", "document editing", "public speaking"],
    "period": ["month", "week", "quarter", "year", "season"],
    "place": ["library", "school", "hotel", "restaurant", "museum", "market"],
    "place_plural": ["towns", "villages", "districts", "neighborhoods", "communities"],
    "product": ["winter coats", "coffee makers", "desk lamps", "travel bags", "notebooks"],
    "project": ["park renovation", "street repair", "library expansion", "harbor cleanup", "market redesign"],
    "public_notice": ["waste collection", "parking rules", "water service", "street cleaning", "library hours"],
    "region": ["coast", "valley", "city", "plain", "mountain area"],
    "room": ["bedroom", "kitchen", "living room", "hallway", "study"],
    "scene": ["city streets", "quiet gardens", "summer markets", "harbor lights", "mountain paths"],
    "season": ["spring", "summer", "autumn", "winter", "holiday"],
    "setting": ["small coastal town", "busy city block", "quiet mountain village", "riverfront neighborhood", "historic district"],
    "strategy": ["defense", "lineup", "passing plan", "training routine", "match schedule"],
    "subject": ["digital photography", "local history", "modern architecture", "public art", "landscape painting"],
    "team": ["football team", "basketball team", "rowing club", "tennis squad", "running group"],
    "time_of_day": ["sunrise", "noon", "midnight", "early morning", "late evening"],
    "time_phrase": ["eight in the morning", "fifteen minutes early", "before sunset", "after lunch", "near noon"],
    "tool": ["screwdriver", "wrench", "hammer", "small drill", "measuring tape"],
    "topic": ["parking rules", "public gardens", "local events", "street lighting", "market hours"],
    "transport": ["train", "bus", "ferry", "tram", "airport shuttle"],
    "venue": ["stadium", "theater", "community hall", "public square", "concert hall"],
    "water_body": ["lake", "river", "harbor", "canal", "bay"],
    "weather": ["clear skies", "light rain", "cool weather", "warm sunshine", "cloudy conditions"],
}


def render(template: str, rng: random.Random) -> str:
    out = template
    for key, values in VALUES.items():
        marker = "{" + key + "}"
        if marker in out:
            out = out.replace(marker, rng.choice(values))
    return out


def contains_forbidden(text: str) -> bool:
    tokens = {
        token.strip(".,;:!?()[]{}\"'").lower()
        for token in text.replace("-", " ").split()
    }
    return bool(tokens & FORBIDDEN_TERMS)


def generate_sentences(count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict[str, str]] = []
    attempts = 0
    while len(rows) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError(f"Unable to generate {count} unique sentences")
        category, template = rng.choice(TEMPLATES)
        text = render(template, rng)
        if text in seen or contains_forbidden(text):
            continue
        seen.add(text)
        rows.append({"category": category, "text": text})
    return rows


def write_jsonl(path: Path, rows: list[dict[str, str]], prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, row in enumerate(rows, 1):
            payload = {
                "noise_id": f"{prefix}_{idx:04d}",
                "category": row["category"],
                "text": row["text"],
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic general-domain easy-noise JSONL banks.")
    parser.add_argument("--output-dir", default="data/noise_text")
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--eval-count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = generate_sentences(args.train_count + args.eval_count, seed=args.seed)
    train_rows = rows[: args.train_count]
    eval_rows = rows[args.train_count :]
    train_texts = {row["text"] for row in train_rows}
    eval_texts = {row["text"] for row in eval_rows}
    overlap = train_texts & eval_texts
    if overlap:
        raise RuntimeError(f"Train/eval overlap detected: {len(overlap)}")

    out_dir = Path(args.output_dir)
    train_path = out_dir / "easy_random_text_train.jsonl"
    eval_path = out_dir / "easy_random_text_eval.jsonl"
    write_jsonl(train_path, train_rows, "easy_train")
    write_jsonl(eval_path, eval_rows, "easy_eval")

    print(
        json.dumps(
            {
                "train_path": str(train_path),
                "eval_path": str(eval_path),
                "train_count": len(train_rows),
                "eval_count": len(eval_rows),
                "overlap": len(overlap),
                "seed": args.seed,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

