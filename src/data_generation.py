import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import random
from src.utils import setup_logging

logger = setup_logging()

def generate_food_dataset(n_samples=50000):
    """
    Generate a synthetic dataset of Ethiopian and European foods with nutritional data, including glucose content, glycemic index, and glycemic load.
    Args:
        n_samples (int): Number of dataset entries.
    Returns:
        pd.DataFrame: Dataset with food names, categories, and nutritional info.
    """
    try:
        # Expanded lists of Ethiopian and European foods with GI ranges
        ethiopian_foods = [
            {"name": "Injera", "category": "Bread", "carb_range": (50, 60), "calorie_range": (200, 250), "protein_range": (4, 7), "fat_range": (1, 3), "gi_range": (50, 57)},
            {"name": "Doro Wat", "category": "Stew", "carb_range": (5, 15), "calorie_range": (150, 200), "protein_range": (10, 15), "fat_range": (8, 12), "gi_range": (40, 50)},
            {"name": "Tibs", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (180, 220), "protein_range": (20, 25), "fat_range": (10, 15), "gi_range": (0, 10)},
            {"name": "Shiro", "category": "Stew", "carb_range": (20, 30), "calorie_range": (120, 160), "protein_range": (8, 12), "fat_range": (5, 8), "gi_range": (45, 55)},
            {"name": "Kitfo", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (200, 250), "protein_range": (18, 22), "fat_range": (15, 20), "gi_range": (0, 10)},
            {"name": "Misir Wat", "category": "Stew", "carb_range": (25, 35), "calorie_range": (100, 140), "protein_range": (6, 10), "fat_range": (3, 6), "gi_range": (50, 60)},
            {"name": "Gomen", "category": "Vegetable", "carb_range": (5, 10), "calorie_range": (50, 80), "protein_range": (2, 4), "fat_range": (1, 3), "gi_range": (30, 40)},
            {"name": "Ayib", "category": "Cheese", "carb_range": (0, 3), "calorie_range": (100, 130), "protein_range": (8, 12), "fat_range": (7, 10), "gi_range": (0, 10)},
            {"name": "Teff Porridge", "category": "Porridge", "carb_range": (40, 50), "calorie_range": (150, 180), "protein_range": (5, 8), "fat_range": (2, 4), "gi_range": (50, 60)},
            {"name": "Fitfit", "category": "Bread", "carb_range": (45, 55), "calorie_range": (180, 220), "protein_range": (4, 7), "fat_range": (2, 5), "gi_range": (55, 65)}
        ]
        
        european_foods = [
            {"name": "Pasta", "category": "Pasta", "carb_range": (65, 75), "calorie_range": (300, 350), "protein_range": (10, 14), "fat_range": (1, 3), "gi_range": (40, 50)},
            {"name": "Croissant", "category": "Pastry", "carb_range": (40, 50), "calorie_range": (350, 400), "protein_range": (6, 9), "fat_range": (20, 25), "gi_range": (65, 75)},
            {"name": "Baguette", "category": "Bread", "carb_range": (50, 60), "calorie_range": (250, 300), "protein_range": (8, 12), "fat_range": (1, 3), "gi_range": (70, 80)},
            {"name": "Pizza", "category": "Main Dish", "carb_range": (30, 40), "calorie_range": (250, 300), "protein_range": (10, 15), "fat_range": (10, 15), "gi_range": (45, 55)},
            {"name": "Roast Beef", "category": "Meat Dish", "carb_range": (0, 5), "calorie_range": (200, 250), "protein_range": (25, 30), "fat_range": (10, 15), "gi_range": (0, 10)},
            {"name": "Mashed Potatoes", "category": "Side Dish", "carb_range": (15, 25), "calorie_range": (100, 140), "protein_range": (2, 4), "fat_range": (3, 6), "gi_range": (80, 90)},
            {"name": "Paella", "category": "Main Dish", "carb_range": (20, 30), "calorie_range": (200, 250), "protein_range": (12, 18), "fat_range": (8, 12), "gi_range": (50, 60)},
            {"name": "Tiramisu", "category": "Dessert", "carb_range": (30, 40), "calorie_range": (300, 350), "protein_range": (5, 8), "fat_range": (15, 20), "gi_range": (50, 60)},
            {"name": "Schnitzel", "category": "Meat Dish", "carb_range": (10, 20), "calorie_range": (250, 300), "protein_range": (20, 25), "fat_range": (12, 18), "gi_range": (30, 40)},
            {"name": "Risotto", "category": "Main Dish", "carb_range": (25, 35), "calorie_range": (200, 250), "protein_range": (8, 12), "fat_range": (6, 10), "gi_range": (60, 70)}
        ]
        
        all_foods = ethiopian_foods + european_foods
        data = {
            "Food_Name": [],
            "Category": [],
            "Carbohydrate_g_per_100g": [],
            "Glucose_g_per_100g": [],
            "Glycemic_Index": [],
            "Glycemic_Load": [],
            "Calories_kcal_per_100g": [],
            "Protein_g_per_100g": [],
            "Fat_g_per_100g": []
        }
        
        logger.info(f"Generating dataset with {n_samples} samples...")
        for _ in range(n_samples):
            food = random.choice(all_foods)
            carb_content = round(random.uniform(food["carb_range"][0], food["carb_range"][1]), 2)
            gi = round(random.uniform(food["gi_range"][0], food["gi_range"][1]), 2)
            # Estimate glucose content
            glucose_content = round(carb_content * (gi / 100), 2)
            # Calculate glycemic load for 100g serving
            glycemic_load = round((carb_content * gi) / 100, 2)
            
            data["Food_Name"].append(food["name"].lower())
            data["Category"].append(food["category"])
            data["Carbohydrate_g_per_100g"].append(carb_content)
            data["Glucose_g_per_100g"].append(glucose_content)
            data["Glycemic_Index"].append(gi)
            data["Glycemic_Load"].append(glycemic_load)
            data["Calories_kcal_per_100g"].append(round(random.uniform(food["calorie_range"][0], food["calorie_range"][1]), 2))
            data["Protein_g_per_100g"].append(round(random.uniform(food["protein_range"][0], food["protein_range"][1]), 2))
            data["Fat_g_per_100g"].append(round(random.uniform(food["fat_range"][0], food["fat_range"][1]), 2))
    
        df = pd.DataFrame(data)
        logger.info("Dataset generated successfully.")
        return df
    
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise

if __name__ == "__main__":
    df = generate_food_dataset()
    df.to_csv("../food_carbohydrate_dataset.csv", index=False)
    logger.info("Dataset saved to '../food_carbohydrate_dataset.csv'")