from google import genai
import time
from tqdm import tqdm
from google.genai import types

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

prompt= """
You are a highly creative and meticulous prompt generator for a cutting-edge text-to-3D diffusion model. Your primary task is to generate **500 unique text prompts**, each describing a **single, distinct, and highly visual 3D object or structured scene element.**

---

### Goal and Expansive Diversity Constraints:

The generated collection of objects must be **hyper-varied** and **maximally diverse**, drawing inspiration from all forms of media, history, and imagination. Ensure the prompts comprehensively cover the following major categories, with rich, descriptive detail:

1.  **Everyday, Tools, and Artifacts:**
    * **Practical:** A perfectly arranged sushi bento box, a complex wind-up clock mechanism, an antique brass telescope.
    * **Relics & Treasures:** A glowing Atlantean crystal, a ceremonial Mayan mask, a cursed dagger encrusted with jewels.
    * Accessories, jewelry, gadgets, household items, musical instruments, sports equipment, clothes, office supplies, toys, etc, etc. Endless possibilties.
2.  **Characters, Creatures, and Figurines:**
    * **Characters from popular culture:**: Examples: Anime characters, Marvel characters such as spider-man, hulk, etc, characters from games such as Ezio Auditore, Lara Croft, Mario, Kratos, cartoon characters, etc., Indian Jones, Sherlock Holmes, Harry Potter, human characters, standard humans such as man, woman, kid, etc.
    * **Fantasy & Sci-Fi:** Intricate elves, biomechanical cyborgs, ethereal spirits, Lovecraftian monsters.
    * **Pop Culture & History:** cinematic creatures in dynamic poses, stylized political figures, classic literary characters, characters from animations, sports,.
    * **Abstract/Stylized:** Chibi characters, low-poly mascots, geometric avatars.
3.  **Architecture, Structures, and Scenics:**
    * **Internal & External:** A collapsing spiral staircase, a sleek Brutalist building facade, an ornate Victorian greenhouse, a subterranean alien throne room, Taj Mahal, Tokyo Tower, Hanging gardens of babylon, sydney opera house.
    * **Specific Styles:** Hyper-realistic, stylized claymation, cel-shaded, vaporwave aesthetic.
4.  **Vehicles and Machinery:**
    * **Operational & Conceptual:** Detailed vintage motorcycles, futuristic flying battleships, abandoned industrial robots, specialized scientific equipment (e.g., a particle accelerator component).
    * **Condition:** Rusted, pristine, battle-damaged, overgrown with moss.
5.  **Organic, Flora, and Fauna:**
    * **Animals:** Photorealistic wildlife (e.g., a snow leopard mid-leap), mythical beasts (e.g., a hydra emerging from water), taxidermy displays.
    * **Plants:** Rare succulents, carnivorous plants, an entire ancient, etc.
    * Various fruits and vegetables, flowers, trees, fungi, etc. Don't do bonsai, we already have many bonsai prompts. Rather explore diverse things.
    * Food items and dishes: gourmet dishes, desserts, beverages, noodles, etc.

This is just a guide-- use your creativity to explore and expand upon these categories, do not limit yourself to them. 
** Choose from absolutely random stuff. Keep a mix of realistic everyday objects/things and creative ones. Focus more on realistic/hyperrealistic. Keep very few futuristic items**
---

### Formatting and Output Rules:

* Generate **exactly 500** unique, highly visual prompts. **DO NOT REPEAT** any prompt.
* The output must contain **only** the sequentially numbered list of prompts. **DO NOT INCLUDE** any introductory text, conversational fillers, or surrounding markdown/code blocks.
* The numbering must start at **1.** and proceed sequentially. Ensure **each prompt is on a new line**.

**Sample Output (Do NOT repeat these exact prompts):**
1. Porcghe 911 Carrera S, hyperrealistic
2. statute of David
3. a sleek, angular neon sign that reads "VOID"
4. an intricate, highly detailed mechanical dragonfly with copper wings
5. a crumbling statue of a griffin perched on a stone pillar
...
500. superhero in a dynamic pose, highly detailed (you can use various superheroes/popular characters/anime characters/game characters)

-----

Note #1: Very important: **Focus on single objects rather than scenes; don't include more than 1 separate objects for example a bat and a ball, only focus on one say bat**. 
Note #2: IMPORTANT: Don't add environment details such as dust, neon, fine particles, etc. Focus on the main object (geometry), don't worry about background/texture too much. 
"""


for i in tqdm(range(200)):
    response = client.models.generate_content(
        model="gemini-2.5-pro", contents=prompt,
        config=types.GenerateContentConfig(
        temperature=0.6
    )
    )
    prompts = "\n"+ response.text
    with open("prompts.txt", "a") as f:
        f.write(prompts)
    time.sleep(1)
