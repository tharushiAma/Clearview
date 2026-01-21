import pandas as pd
import os

def augment_data(project_dir):
    # ==========================================
    # PART 1: TRAINING DATA (Seen by Model)
    # ==========================================
    train_price_neg = [
        "The price is absolutely ridiculous for such a small bottle.",
        "Way too expensive for the quality.",
        "I can't believe I paid this much for a drugstore quality lipstick.",
        "Not worth the money at all.",
        "Overpriced and underdelivers.",
        "Cost way too much considering it fades in an hour.",
        "For this price, I expected much better packaging.",
        "A total rip-off.",
        "I wouldn't buy this again unless the price drops significantly.",
        "Too pricey for what you get.",
        "The cost is prohibitive for everyday use.",
        "Extremely expensive.",
        "Waste of money.",
        "I regret spending so much on this.",
        "You're paying for the brand name, not the quality.",
        "Cheap product with a luxury price tag.",
        "I could buy three better lipsticks for the price of this one.",
        "The value for money is terrible.",
        "Pricey not worth it.",
        "Expensive garbage.",
        "High cost, low quality.",
        "Not reasonable at all.",
        "I felt cheated by the price.",
        "Should be half the cost.",
        "Exorbitant price.",
        "Definitely not worth $40.",
        "Price point is too high.",
        "Cost is unacceptably high.",
        "Overcharged for this.",
        "Pricey and disappointing.",
        "Not a good value.",
        "Way over budget for a lip balm.",
        "Too much money.",
        "Price is a joke.",
        "Can't justify the cost.",
        "Ridiculously expensive.",
        "Sad that it costs so much.",
        "Price barrier is high.",
        "Costs an arm and a leg.",
        "Poor value.",
        "Financial mistake to buy this.",
        "Overpriced hype.",
        "Costly disappointment.",
        "Price is steep.",
        "Price tag is shocking.",
        "Not worth the high price.",
        "Way too costly.",
        "Expensive mistake.",
        # NEW DIVERSE PATTERNS (Subtle/Comparative/Quantity-based)
        "Too expensive for the amount you get.", # Logic: Quantity/Price ratio
        "The price is a bit high for a drugstore brand.", # Logic: Category comparison
        "Good product but the cost adds up.",
        "I like it but I can't justify repurchasing at this price.",
        "It's on the pricier side.",
        "A bit steep for what it is.",
        "The tube is tiny for how much it costs.",
        "Not the best bang for your buck.",
        "I usually buy cheaper alternatives that work just as well.",
        "Great quality but hurts the wallet.",
        "I'd buy it more often if it wasn't so dear.",
        "The price point is slightly off.",
        "Paying for the name effectively.",
        "Compared to [Brand X], this is overpriced.",
        "It's decent, but not $30 decent.",
        "Hard to swallow the price tag.",
        "The quantity is misleading for the price.",
        "For a budget brand, this is expensive.",
        "It's an investment, maybe too much of one.",
        "Pricey given how fast it runs out."
        # Mixed Training Examples
        "I absolutely love the shade and the texture is creamy, but honestly, $45 is ridiculous for such a small tube.",
        "Beautiful color and stays on well, but I can't justify the high price tag.",
        "The formula is moisturizing and the pigment is great, however, it is way too expensive for what it is.",
        "Gorgeous packaging and lovely scent, but not worth the money at all.",
        "Best red lipstick I've owned, but the cost is prohibitive.",
        "It applies smoothly and looks fantastic, but frankly, it's overpriced.",
        "Great coverage and long-lasting, but I felt cheated by the price.",
        "I adore the matte finish, but for this price, I expected more product.",
        "Perfect everyday color, but purely based on value for money, I wouldn't buy it again.",
        "The shipping was fast and the color is true to match, but this is just expensive garbage.",
        "Love the brand and the quality is decent, but the price point is too high.",
        "It's a nice lipstick, vibrant and soft, but definitively not worth $40.",
        "The staying power is incredible, but the price makes it a onetime purchase only.",
        "Nice texture, no smell, but cost is unacceptably high.",
        "The pigment is to die for, but the price is a joke.",
        "Packaging is cute and color is pop, but it costs an arm and a leg.",
        "Super hydrating and smells like vanilla, but it's an expensive mistake.",
        "I like how it feels on the lips, but 30 dollars is too much.",
        "Really good quality, but you are paying for the brand name essentially.",
        "Vibrant color pay-off, sucks that it costs so much."
    ]



    train_packing_neg = [
        "The packaging feels cheap and flimsy.",
        "The cap fell off in my bag and ruined everything.",
        "Box arrived crushed.",
        "The lipstick mechanism broke after two uses.",
        "Terrible packaging, the lid doesn't stay on.",
        "Container cracked immediately.",
        "Cheap plastic casing.",
        "The formatting and design look tacky.",
        "Packaging is not sturdy.",
        "Came with no safety seal, which is concerning.",
        "The tube feels very light and cheap.",
        "Difficult to open.",
        "The twist-up mechanism is jammed.",
        "Leaked all over the package.",
        "Box was torn.",
        "The lipstick bullet fell out of the holder.",
        "Packaging looks nothing like the picture.",
        "Very poor quality container.",
        "Lid is loose.",
        "Packaging is wasteful.",
        "Arrived broken.",
        "The case scratched easily.",
        "Looks counterfeit due to bad printing on the box.",
        "Sloppy packaging.",
        "The box was damp and falling apart.",
        "Unacceptable condition upon arrival.",
        "The wrapper was sticky.",
        "Packaging design is ugly.",
        "Felt like a dollar store package.",
        "Broken seal.",
        "The tube is defective.",
        "Cannot retract the lipstick.",
        "Packaging failure.",
        "Worst packaging ever.",
        "Damaged box.",
        "Container fell apart.",
        "Cheap wrapper.",
        "Lid cracks easily.",
        "Poorly designed tube.",
        "Cardboard box was ripped.",
        "Plastic seal was missing.",
        "Mechanism stuck.",
        "Package looked old and worn.",
        "Dirty packaging.",
        "The wand broke off inside.",
        "Leaky container.",
        "Melted due to bad insulation in shipping.",
        # Mixed Training Examples
        "The color is stunning and delivery was fast, but the box arrived completely crushed.",
        "I love the lipstick itself, it's very smooth, but the cap fell off in my bag immediately.",
        "Great product inside, but the packaging feels cheap and flimsy.",
        "Fast shipping and great customer service, sadly the lipstick mechanism broke after one use.",
        "The shade is exactly what I wanted, but the container cracked immediately.",
        "Beautiful finish and long-wearing, but the tube feels very light and cheap.",
        "Smells delicious and looks great, but it came with no safety seal which is concerning.",
        "The lipstick is high quality, but the packaging design is ugly and tacky.",
        "Perfect nude shade, but the twist-up mechanism is jammed.",
        "I'm happy with the color, but the box was torn and looked old.",
        "Amazing formula, but the lid is loose and won't stay customized.",
        "Best matte lipstick I've tried, but the case scratched easily.",
        "The product itself is 5 stars, but the packaging is 1 star.",
        "Love the texture, hate the cheap plastic casing.",
        "It arrived quickly, but the lipstick bullet had fallen out of the holder.",
        "Great value for money product-wise, but the packaging is wasteful.",
        "Highly pigmented and creamy, but the wrapper was sticky.",
        "Good staying power, but the container fell apart in my hand.",
        "Nice color, but the lid cracks easily.",
        "The lipstick is luxurious, the packaging feels like a dollar store item."
    ]

    # ==========================================
    # PART 2: VALIDATION DATA (New & Unseen)
    # ==========================================
    val_price_neg = [
        # Pure Negative
        "I honestly think it is too expensive for a drugstore brand.",
        "The color is nice but $35 is robbery.",
        "I would buy it again if it wasn't so pricey.",
        "Overpriced compared to simpler products.",
        "It costs way too much.",
        "Great lipstick, terrible price.",
        "Not worth the high cost.",
        "Spending 40 bucks on this was a mistake.",
        "The value simply isn't there.",
        "Way too much money for such a tiny tube.",
        "Price is the only downside.",
        "Wish it was cheaper.",
        "Good product but bad price.",
        "Too heavy on the wallet.",
        "Not budget friendly.",
        # Mixed
        "I like the texture, but the price is too high.",
        "The color payoff is amazing, sadly it is ultra expensive.",
        "Lasts all day, but costs a fortune.",
        "My favorite shade, but I can't afford to buy it often.",
        "Five stars for quality, one star for price."
    ]

    val_packing_neg = [
        # Pure Negative
        "The box was completely smashed when I got it.",
        "Lid doesn't click shut.",
        "Pump dispenser is broken.",
        "Label was peeling off.",
        "Glass bottle shattered efficiently.",
        "Applicator fell apart.",
        "Came in a dirty envelope.",
        "Safety seal was already broken.",
        "Cap is cracked.",
        "Tube feels sticky.",
        # Mixed
        "Great color, but the lid is loose.",
        "Love the smell, but the packaging feels cheap.",
        "Product is okay, but the box was damaged.",
        "Fast shipping, but the container leaks.",
        "Nice lipstick, terrible tube design."
    ]

    # Helper function to create dataframe
    def create_augmented_df(original_path, price_texts, packing_texts, output_path):
        print(f"Loading {original_path}...")
        df = pd.read_parquet(original_path)
        print(f"Original size: {len(df)}")
        
        new_rows = []
        for t in price_texts:
            new_rows.append({"text_clean": t, "data": t, "stayingpower": "na", "texture": "na", "smell": "na", "price": "negative", "colour": "na", "shipping": "na", "packing": "na", "signature": "aug_val_price"})
        for t in packing_texts:
            new_rows.append({"text_clean": t, "data": t, "stayingpower": "na", "texture": "na", "smell": "na", "price": "na", "colour": "na", "shipping": "na", "packing": "negative", "signature": "aug_val_packing"})
            
        new_df = pd.DataFrame(new_rows)
        aug_df = pd.concat([df, new_df], ignore_index=True)
        
        print(f"Augmented size: {len(aug_df)}")
        aug_df.to_parquet(output_path)
        print(f"Saved to {output_path}")

    # Generate Training Augmentation
    create_augmented_df(
        os.path.join(project_dir, "data", "splits", "train.parquet"),
        train_price_neg,
        train_packing_neg,
        os.path.join(project_dir, "data", "splits", "train_augmented.parquet")
    )

    # Generate Validation Augmentation
    create_augmented_df(
        os.path.join(project_dir, "data", "splits", "val.parquet"),
        val_price_neg,
        val_packing_neg,
        os.path.join(project_dir, "data", "splits", "val_augmented.parquet")
    )

if __name__ == "__main__":
    augment_data(".")
