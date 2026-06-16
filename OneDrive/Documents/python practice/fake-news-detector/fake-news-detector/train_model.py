import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

real_news = [
    "Scientists discover new vaccine effective against multiple strains of influenza.",
    "Government announces new infrastructure bill worth 1.2 trillion dollars.",
    "NASA successfully launches Mars rover to study the planet's geology.",
    "Federal Reserve raises interest rates by 0.25 percent amid inflation concerns.",
    "World leaders meet at UN summit to discuss climate change action plans.",
    "Apple unveils new iPhone with improved battery life and camera features.",
    "Study shows regular exercise reduces risk of heart disease by 30 percent.",
    "Electric vehicle sales surpass one million units for the first time.",
    "New study links air pollution to increased rates of respiratory illness.",
    "Stock market reaches all-time high following strong economic report.",
    "Scientists find evidence of water ice on the surface of Mercury.",
    "Unemployment rate drops to lowest level in two decades.",
    "Researchers develop biodegradable plastic alternative from seaweed.",
    "Supreme Court rules on landmark privacy case involving digital data.",
    "International team achieves breakthrough in nuclear fusion energy research.",
    "Global temperatures in 2023 were the highest ever recorded in history.",
    "SpaceX successfully lands reusable rocket for the twentieth time.",
    "Health officials recommend updated COVID booster for high-risk groups.",
    "Tech giants face antitrust scrutiny from European Union regulators.",
    "Astronomers detect gravitational waves from black hole merger event.",
    "New cancer drug shows 80 percent success rate in clinical trials.",
    "Congress passes bipartisan cybersecurity bill to protect infrastructure.",
    "WHO declares end to public health emergency after disease containment.",
    "Renewable energy surpasses coal in electricity generation for the first time.",
    "Scientists sequence genome of ancient mammoth found in Siberian permafrost.",
    "The president signed the new education reform bill into law today.",
    "Oil prices fall sharply after OPEC nations agree to increase production.",
    "Breakthrough battery technology could double electric car range.",
    "New satellite launched to monitor deforestation in the Amazon rainforest.",
    "Study confirms Mediterranean diet reduces risk of dementia significantly.",
    "City council approves new affordable housing development downtown.",
    "Measles outbreak contained after rapid vaccination campaign in region.",
    "Major earthquake strikes off the coast of Japan, tsunami warning issued.",
    "New trade agreement signed between the US and European Union.",
    "Doctors perform first successful whole-eye transplant surgery on veteran.",
    "Global internet outage caused by undersea cable damage near coast.",
    "Researchers create AI model that can predict earthquakes 72 hours ahead.",
    "Central bank governor warns of recession risk if inflation persists.",
    "New telescope captures most detailed image of a distant galaxy cluster.",
    "Scientists warn of accelerating glacier melt due to rising temperatures.",
    "Parliament votes to raise minimum wage for the first time in six years.",
    "Archaeologists uncover 3,000-year-old city beneath Egyptian desert sands.",
    "Airlines report record passenger numbers for the summer travel season.",
    "Rare genetic disorder treated successfully with CRISPR gene editing.",
    "UN peacekeeping forces deployed to conflict zone to protect civilians.",
    "Scientists identify new species of deep-sea fish near Mariana Trench.",
    "Hospital study finds hand hygiene reduces infections by 40 percent.",
    "New bridge connecting two major cities opens ahead of schedule.",
    "World Bank provides emergency funding to nations affected by drought.",
    "Astronomers confirm existence of planet outside our solar system with water.",
]

fake_news = [
    "SHOCKING: Secret government program turns tap water into mind control drug!",
    "BREAKING: Scientists CONFIRM aliens have landed and government is hiding it!",
    "Bill Gates admits microchips are inside COVID vaccines to track everyone.",
    "The earth is actually flat, NASA has been lying to us for 60 years confirmed.",
    "Drinking bleach cures cancer, doctors don't want you to know this secret!",
    "5G towers are spreading viruses and making people sick, whistleblower reveals.",
    "Celebrities are actually lizard people, insider reveals shocking truth.",
    "Moon landing was faked in Hollywood studio, leaked documents prove it.",
    "Chemtrails are poison sprayed by government to control the population.",
    "Obama born in Kenya, new documents surface proving birth certificate is fake.",
    "Deep state controls all elections using secret voting machine hack.",
    "Doctors are suppressing miracle cure for all diseases to make money.",
    "Fluoride in water is government plot to make citizens docile and obedient.",
    "New world order plotting to eliminate 90 percent of world population soon.",
    "Hollywood elites run secret underground child trafficking network exposed!",
    "Man grows back amputated arm using ancient Chinese herb secret technique.",
    "Vaccines cause autism, massive study of 500 families proves it conclusively.",
    "George Soros funding antifa terrorists to start civil war in America.",
    "Reptilian shapeshifters infiltrate government at highest levels, proof inside.",
    "Miracle berry from Amazon cures diabetes in three days doctors are furious.",
    "QAnon insider reveals deep state plan to arrest all conservative leaders.",
    "COVID was created in lab and released intentionally by China to destroy USA.",
    "Area 51 has real alien spaceship that government reverse-engineered for tech.",
    "Eating microwave food for 30 days causes permanent brain damage scientifically.",
    "Prince Charles secretly admitted on tape he ordered Princess Diana's death.",
    "Scientists prove prayer cures cancer better than chemotherapy every time.",
    "Hollywood is using subliminal messages in movies to brainwash children.",
    "This simple trick eliminates all debt instantly and banks hate it so much!",
    "Giant ancient pyramids discovered on Mars surface prove advanced civilization.",
    "Secret society of bankers controls every world government behind the scenes.",
    "Man cures HIV using only coconut oil in 7 days, hospitals cover it up.",
    "Pilot admits planes are spraying chemicals to cause widespread memory loss.",
    "Shadow government using HAARP to control weather and cause natural disasters.",
    "New study proves eating sugar cures Alzheimer's disease within 2 months.",
    "Elon Musk is actually an AI robot built by the deep state to control tech.",
    "Ancient giants once walked the earth and the Smithsonian is hiding bones.",
    "Scientist fired for proving evolution is completely false and a massive lie.",
    "Secret cure for all cancer found in 1930s but was suppressed by Big Pharma.",
    "Politician caught on tape admitting to worshipping Satan at private ceremony.",
    "NASA confirms the sun is actually cold and made entirely of dark matter.",
    "Man lives for 200 years by drinking special water from hidden mountain spring.",
    "Underground bunker network built for elites to survive planned apocalypse.",
    "Mainstream media controlled by 6 families who decide what you are allowed to think.",
    "Scientists discover human brains have secret third hemisphere no one talks about.",
    "Government spraying autism-causing chemicals over suburban neighborhoods at night.",
    "Leaked emails show top scientists faking all climate change data for decades.",
    "This plant extract kills all viruses in 10 minutes big pharma buys and destroys.",
    "Children are being hypnotized by TikTok algorithm controlled by foreign agents.",
    "The Vatican is hiding a library of alien technology beneath St. Peter's Basilica.",
    "Whistleblower exposes secret that the moon is actually a giant space station.",
]

texts  = real_news + fake_news
labels = ["REAL"] * len(real_news) + ["FAKE"] * len(fake_news)

df = pd.DataFrame({"text": texts, "label": labels})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset built: {len(df)} articles")

X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

pac = PassiveAggressiveClassifier(max_iter=50, random_state=42, C=0.5)
pac.fit(X_train_tfidf, y_train)

y_pred = pac.predict(X_test_tfidf)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

os.makedirs("model", exist_ok=True)

with open("model/model.pkl",      "wb") as f: pickle.dump(pac,  f)
with open("model/vectorizer.pkl", "wb") as f: pickle.dump(tfidf, f)

print("Model saved!")
print("Vectorizer saved!")
print("Now run: python app.py")