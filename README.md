# Recruitment Need Analysis Tool

## Setup

1. Install the dependencies:

```bash
pip install -r requirements.txt
```

2. Start the Streamlit interface:

```bash
streamlit run "Recruitment_Need_Analysis_Tool.py"
```

The **SKILLS** step suggests additional hard and soft skills via OpenAI and
presents them as selectable buttons. Chosen suggestions are stored in your
profile. A dynamic salary chart visualises the impact of your selected skills.
Skills are grouped into language skills, key competencies and other
requirements for a cleaner layout.

The **BENEFITS** step generates benefit suggestions based on the job title, the
company location and typical competitor offerings. Each suggestion appears as a
button that you can toggle to add it to your benefit list.

The **COMPANY & DEPARTMENT** step now groups company information in a clearer
layout and includes a *Team & Culture Context* expander. Optional fields such as
Tech Stack and Team Challenges offer a **Generate Ideas** button that fetches AI
suggestions you can insert directly.

## Wizard Steps

The wizard collects data in the following order:

1. BASIC
2. COMPANY & DEPARTMENT – company and team details
3. ROLE & TASKS – responsibilities and key tasks
4. SKILLS
5. BENEFITS
6. INTERVIEW
7. SUMMARY – includes Ideal Candidate Profile and Target Industries fields

In the final **SUMMARY** step you can generate a job advertisement, an interview
guide, a Boolean search string and a draft contract. New buttons also let you
estimate the salary range and calculate the total annual compensation based on
selected benefits.
