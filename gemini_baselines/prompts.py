"""Prompt templates of the original paper, copied verbatim.

Every template below is a byte-for-byte copy of the corresponding template in
``framework/`` and ``evaluation.py`` (including the unusual indentation and the
truncated in-context examples).  Keeping them identical is what makes the
baselines a fair comparison -- do **not** "improve" these prompts.

Source map
----------
``DIRECT_GENERATION``          framework/generation-based/direct_generation.py
``TWOSTAGE_GET_CANDIDATE``     framework/generation-based/twostage_generation.py
``CANDIDATE_LIST_REPAIR``      framework/generation-based/twostage_generation.py
``TWOSTAGE_CHOICE``            framework/generation-based/twostage_generation.py
``EVENT_ANALYSIS``             framework/generation-based/summary_generation.py
``SUMMARY_GET_CANDIDATE``      framework/generation-based/summary_generation.py
``SUMMARY_CHOICE``             framework/generation-based/summary_generation.py
``RETRIEVAL_CHOICE``           framework/retrieval-based/twostage_retrieval.py
``REFLECTION_GET_CANDIDATE``   framework/generation-based/reflection_generation.py
``REFLECTION_CHOICE``          framework/generation-based/reflection_generation.py
``REFLECTION_WARMUP``          framework/generation-based/reflection_generation.py
``EVAL_EXTRACT_FEATURES*``     evaluation.py
``EVAL_ABSTRACT_SIMILARITY``   evaluation.py
"""

# --------------------------------------------------------------------------
# Direct Generation  (direct_generation.py :: get_analogy)
# --------------------------------------------------------------------------
DIRECT_GENERATION = '''You are a historical analogy bot. For input events, your goal is to find the event that best fits the analogy. Here is a case:

    ==== case
    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    Historical Analogies Events:
    Spanish flu

    ==== Answer the following questions using the format given above
    Input Event:
    {event}
    Historical Analogies Events:
    '''

# --------------------------------------------------------------------------
# Two-stage Generation  (twostage_generation.py)
# --------------------------------------------------------------------------
TWOSTAGE_GET_CANDIDATE = '''
    You are a historical analogy robot.
    For input events, please consider the summary, background, process and results, output 10 historical events that are similar in many aspects above, and return them in list format.

    ===== The following is an example:
    Input Event:
    2019–20 coronavirus pandemic
    The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history.\nCOVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    The 10 historical events that are similar with input:
    ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa","SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague pandemics"]

    ===== question
    Input Event:
    {event}
    The 10 historical events that are similar with input:
    '''

CANDIDATE_LIST_REPAIR = '''Output the given analog historical events in [Python list format], which means enclosed by "[]", each item is framed by double quotes (") and items are connected by commas(,).
        The following is an output example:
        ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa",
            "SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague's pandemics"]
        Only analog historical events need to be output, and all other information is ignored.
        Use single quotes(') instead of double quotes(") in event names. And events need to be enclosed in double quotes: like "A"B" and "A"B is not allowed item

        Output the historical analogy events in the text according to the above format：
        {text}

        Python list analog historical events:
        '''

TWOSTAGE_CHOICE = '''You are an analogy robot. For the input event and the historical event used for selection, your goal is to find the best event that can be used for analogies. Here is a case:

    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    Optional Historical Events:
    2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
    Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry.The source of the illness remained a mystery until the cases were mapped out using epidemiological method. The origin was found to be the polluted chalk water well at Addington, London, which supplied water to up to one-fifth of the area that is now the London Borough of Croydon. Coupled with issues around the co-operation between the medical officers and the administrators of the Borough, three coincidental events were blamed; changes to the well structure by repair work, the employment of a new workman who was an unwitting carrier of typhoid, and failure to chlorinate the water.
    Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
    Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. Historians do not fully agree on its starting and ending points, but the period is generally considered to span the 1947 Truman Doctrine to the 1991 Dissolution of the Soviet Union. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945. Aside from the nuclear arsenal development and conventional military deployment, the struggle for dominance was expressed via indirect means such as psychological warfare, propaganda campaigns, espionage, far-reaching embargoes, rivalry at sports events and technological competitions such as the Space Race.

    Among the options, the most appropriate one to use as an analogy for 2019–20 coronavirus pandemic is Spanish flu

    Input Event:
    {input_event}
    Optional Historical Events:
    {candidate_events}

    Among the options, the most appropriate one to use as an analogy for {input_name} is '''

# --------------------------------------------------------------------------
# Generation with Summarizing  (summary_generation.py)
# --------------------------------------------------------------------------
EVENT_ANALYSIS = '''
    You are an event summary robot. For the long event description input, please combine your knowledge and summarize it into four parts: summary, background, process and result. The summary should be concise, with each parts consisting of only one sentence and no more than 100 words.
    The following is an example:

    Input Event:
    September 11 attacks: The September 11 attacks, commonly known as 9/11,[f] were four coordinated Islamist suicide terrorist attacks carried out by al-Qaeda against the United States in 2001...
    Output:
    1. Summary: The September 11 attacks, orchestrated by al-Qaeda, involved four coordinated terrorist hijackings, resulting in the deadliest terrorist attack in history with 2,977 fatalities.
    2. Background: Al-Qaeda, led by Osama bin Laden, targeted the U.S. due to its support of Israel, military presence in Saudi Arabia, and sanctions against Iraq.
    3. Process: On September 11, 2001, 19 terrorists hijacked four planes, crashing two into the World Trade Center in New York, one into the Pentagon, and the fourth in Pennsylvania after passengers revolted.
    4. Result: The attacks led to the U.S. launching the War on Terror, including invasions of Afghanistan and Iraq, substantial global anti-terrorism legislation, and long-term impacts on global security and economy.

    Input Event: {event}
    Output:
    '''

SUMMARY_GET_CANDIDATE = '''
    You are a historical analogy robot.
    For input events, please consider the summary, background, process and results, output 10 historical events that are similar in many aspects above, and return them in list format.
    The following is an example:

    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
    Output: ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa","SARS outbreak","H1N1 influenza pandemic","MERS outbreak","Cholera pandemics","Plague pandemics"]

    Input Event:
    {event}
    Output: '''

SUMMARY_CHOICE = '''You are an analogy robot. For the input event and the historical event used for selection, your goal is to find the best event that can be used for analogies. Here is a case:

    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
    Optional Historical Events:
    2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
    Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...
    Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
    Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945...
    Historical Analogies Events:
    Spanish flu

    Input Event:
    {input_event}
    Optional Historical Events:
    {candidate_events}
    Historical Analogies Events:
    '''

# --------------------------------------------------------------------------
# Two-stage Retrieval  (twostage_retrieval.py :: llm_choice)
# --------------------------------------------------------------------------
RETRIEVAL_CHOICE = '''You are an analogy robot. For the input event and the historical event used for selection, your goal is to find the best event that can be used for analogies. Here is a case:

    ==== case
    Input Event:
    2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2. The novel virus was first identified in Wuhan, China, in December 2019; a lockdown in Wuhan and other cities in Hubei province failed to contain the outbreak, and it spread to other parts of mainland China and around the world. The World Health Organization declared a Public Health Emergency of International Concern on 30 January 2020, and a pandemic on 11 March 2020. Since 2021, variants of the virus have emerged and become dominant in many countries, with the Delta, Alpha and Beta variants being the most virulent. As of 30 September 2021, more than 233 million cases and 4.77 million deaths have been confirmed, making it one of the deadliest pandemics in history. COVID-19 symptoms range from unnoticeable to life-threatening. Severe illness is more likely in elderly patients and those with certain underlying medical conditions. The disease transmits when people breathe in air contaminated by droplets and small airborne particles.
    Optional Historical Events:
    2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
    Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry.The source of the illness remained a mystery until the cases were mapped out using epidemiological method. The origin was found to be the polluted chalk water well at Addington, London, which supplied water to up to one-fifth of the area that is now the London Borough of Croydon. Coupled with issues around the co-operation between the medical officers and the administrators of the Borough, three coincidental events were blamed; changes to the well structure by repair work, the employment of a new workman who was an unwitting carrier of typhoid, and failure to chlorinate the water.
    Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
    Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. Historians do not fully agree on its starting and ending points, but the period is generally considered to span the 1947 Truman Doctrine to the 1991 Dissolution of the Soviet Union. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945. Aside from the nuclear arsenal development and conventional military deployment, the struggle for dominance was expressed via indirect means such as psychological warfare, propaganda campaigns, espionage, far-reaching embargoes, rivalry at sports events and technological competitions such as the Space Race.
    Historical Analogies Events:
    Spanish flu

    ==== Answer the following questions using the format given above
    Input Event:
    {input_event}
    Optional Historical Events:
    {candidate_events}
    Historical Analogies Events:
    '''

# --------------------------------------------------------------------------
# Self-reflection Framework  (reflection_generation.py)
# --------------------------------------------------------------------------
# The original uses a LangChain LLMChain with ConversationBufferMemory
# (ai_prefix="Output", human_prefix="Input").  ``{chat_history}`` is filled with
# the serialised conversation so far; see reflection_generation.py for the
# equivalent memory handling.
REFLECTION_GET_CANDIDATE = """ You're a robot for getting historical analogies events. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
For input events, please consider the summary, background, process and results, and output 5 historical events that are similar in many aspects above, and return them in list format.
If there is any reflection, please modify the recommended events based on the reflection.
The following is an example:

Input Event:
2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...
Output: ["Spanish flu pandemic","Asian flu pandemic","Hong Kong flu pandemic","AIDS pandemic","Ebola outbreak in West Africa"]

{chat_history}

{input_type}:
{input}
Output:
"""

REFLECTION_WARMUP = '''You are a historical analogy reflection robot. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
        For the input event and the candidate event set, please make a comparison, reflect on the shortcomings of the candidate set, and make suggestions for obtaining a better analogous candidate set. Suggestions should be succinct and concise, with a single sentence indicating the direction of change for the candidate set.
        Here is a example:

        == example
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...

        Optional Historical Events:
        2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
        Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...

        Thought:
        The 2019–20 coronavirus pandemic is a global epidemic, so the themes of 2022 South Asian floods are completely different. The Croydon typhoid outbreak of 1937 was smaller in scope, while the 2019–20 coronavirus pandemic were global influenza pandemics, so there is no suitable analogy here and I need to reflect.

        Reflection:
        Candidate events need to focus on the epidemic and its impact on a global scale.

        ==== question
        Input Event:
        {input_event}

        Optional Historical Events:
        {candidate_events}

        Thought:
        {thought}'''

REFLECTION_CHOICE = '''You are a historical analogy robot. Historical Analogy is comparsion of a known past event or person with a contemporary but unfamiliar event or person in order to identify common aspects between the two.
        For the input event and the candidate event set used for selection, your goal is to find a most suitable event that can be used for historical analogies, which means the two events are similar in causes, processes, results, etc. If the events in the candidate set are not appropriate or better analogies may exist, you should reflect on the shortcomings of these events in the analogies, pointing out the desired focus of the analogies to help find a new candidate set of events.
        Here are two case:

        ==== case 1
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...

        Optional Historical Events:
        2022 South Asian floods: From January to October 2022, excessive rainfall and widespread monsoon flooding occurred in the South Asian countries of Afghanistan, Bangladesh, India, Nepal, Pakistan, and Sri Lanka. It has become the region's deadliest floods since 2020, with over 3,700 people dead.
        Croydon typhoid outbreak of 1937: The Croydon typhoid outbreak of 1937, also known as the Croydon epidemic of typhoid fever, was an outbreak of typhoid fever in Croydon, Surrey, now part of London, in 1937. It resulted in 341 cases of typhoid, and it caused considerable local discontent leading to a media campaign and a public inquiry...

        Thought:
        The 2019–20 coronavirus pandemic is a global epidemic, so the themes of 2022 South Asian floods are completely different. The Croydon typhoid outbreak of 1937 was smaller in scope, while the 2019–20 coronavirus pandemic were global influenza pandemics, so there is no suitable analogy here and I need to reflect.

        Reflection:
        Candidate events need to focus on the epidemic and its impact on a global scale.

        ==== case 2
        Input Event:
        2019–20 coronavirus pandemic: The COVID-19 pandemic, also known as the coronavirus pandemic, is an ongoing global pandemic of coronavirus disease 2019 caused by severe acute respiratory syndrome coronavirus 2...

        Optional Historical Event:
        Spanish flu: The 1918–1920 flu pandemic, also known as the Great Influenza epidemic or by the common misnomer Spanish flu, was an exceptionally deadly global influenza pandemic caused by the H1N1 influenza A virus. The earliest documented case was March 1918 in the state of Kansas in the United States, with further cases recorded in France, Germany and the United Kingdom in April. Two years later, nearly a third of the global population, or an estimated 500 million people, had been infected in four successive waves. Estimates of deaths range from 17 million to 50 million,[6] and possibly as high as 100 million, making it one of the deadliest pandemics in history.
        Cold War: The Cold War was a period of geopolitical tension between the United States and the Soviet Union and their respective allies, the Western Bloc and the Eastern Bloc, which began following World War II. The term cold war is used because there was no large-scale fighting directly between the two superpowers, but they each supported major regional conflicts known as proxy wars. The conflict was based around the ideological and geopolitical struggle for global influence by these two superpowers, following their temporary alliance and victory against Nazi Germany in 1945...

        Thought:
        The Cold War has nothing to do with the epidemic. The Spanish flu is also an epidemic and has had a great impact in Europe, so it is a qualified analogy for the 2019–20 coronavirus pandemic.

        Final Answer:
        Spanish flu

        ==== question
        Input Event:
        {input_event}

        Optional Historical Events:
        {candidate_events}

        Thought:
        {thought}'''

# --------------------------------------------------------------------------
# Evaluation  (evaluation.py)
# --------------------------------------------------------------------------
EVAL_EXTRACT_FEATURES = '''
        You are an event summary robot. For the long event description input, please combine your knowledge and summarize it into four parts: summary, background, process and result. The summary should be concise, with each parts consisting of only one sentence and no more than 100 words.
        The following is an example:

        Input Event:
        September 11 attacks: The September 11 attacks, commonly known as 9/11,[f] were four coordinated Islamist suicide terrorist attacks carried out by al-Qaeda against the United States in 2001...
        Output:
        1. Summary: The September 11 attacks, orchestrated by al-Qaeda, involved four coordinated terrorist hijackings, resulting in the deadliest terrorist attack in history with 2,977 fatalities.
        2. Background: Al-Qaeda, led by Osama bin Laden, targeted the U.S. due to its support of Israel, military presence in Saudi Arabia, and sanctions against Iraq.
        3. Process: On September 11, 2001, 19 terrorists hijacked four planes, crashing two into the World Trade Center in New York, one into the Pentagon, and the fourth in Pennsylvania after passengers revolted.
        4. Result: The attacks led to the U.S. launching the War on Terror, including invasions of Afghanistan and Iraq, substantial global anti-terrorism legislation, and long-term impacts on global security and economy.

        Input Event: {event}
        Output:
        '''

EVAL_EXTRACT_FEATURES_WITH_EXAMPLE = '''
        You are an event summary robot. For the long event description input, please combine your knowledge and summarize it into four parts: summary, background, process and result. The summary should be concise, with each parts consisting of only one sentence and no more than 100 words.
        The following is an example:

        Input Event:
        {event_name}: {event_intro}
        Output:
        1. Summary: {event_summary}
        2. Background: {event_background}
        3. Process: {event_process}
        4. Result: {event_result}

        Input Event: {event}
        Output:
        '''

EVAL_ABSTRACT_SIMILARITY = ''' You are a sentence-level analogy scoring robot. For the two input texts, please judge the quality of the analogy and give it a score (1-4). It should be noted that the quality of an analogy only focuses on the abstract-level similarity of descriptions, not the surface similarity of descriptions. For example, in a good analogy, two descriptions may belong to the same topic and express similar general situations, but they may not necessarily be the same specific process or description.

    ## Grading
    1 point: The description belongs to a completely different topic or field, has no connection, and cannot be compared.
    2 points: The descriptions belong to the same general theme, but the general situation or aspect expressed is significantly different, and the quality of the analogy is low.
    3 points: The descriptions belong to the same topic and express similar general situations, but are somewhat different in details or focus. This is an acceptable analogy.
    4 points: The descriptions belong to exactly the same topic, the general situation expressed is highly similar, the concepts and key points are highly overlapping, and it is a good analogy.
    In addition, there are several points to note:
    1. [Self-analogy is bad!!!]. Similarly, if one description overwrites another description, it is also a bad analogy.
    2. The quality of an analogy is only affected by abstract-level similarity and the similarity or identity of entities does not affect the quality of the analogy. For example, "The United States attacked Japan" and "The United States helped Japan" are completely incomparable; while "The United States attacked Japan" and "Germany invaded France" are good analogies.

    ## The following is two case:
    Case Description 1: On September 11, 2001, 19 terrorists hijacked four planes, crashing them into the World Trade Center, the Pentagon, and a field in Pennsylvania After a passenger revolt.
    Case Description 2: On December 7, 1941, 353 Japanese aircraft attacked Pearl Harbor, damaging or sinking eight battleships and destroying over 180 U.S. aircraft.
    Score: 3

    Case Description 1: The spillover of the Syrian Civil War had significant impacts in the Arab world and beyond, leading to a wider regional conflict and the rise of the Islamic State of Iraq and the Levant.\n\n
    Case Description 2: The Revolutions of 1989 were a series of political changes that led to the end of communist rule in Central and Eastern Europe, marking the end of the Cold War.\n\n
    Score: 2

    ## Question
    Description 1: {text1}
    Description 2: {text2}
    Score:
    '''
