def diet_planner(health_goal,medical_conditions,fitness_routines,preferences):

    from secret_key import googleapi_key
    import os
    #os.environ['COHERE_API_KEY'] = cohereapi_key
    os.environ['GOOGLE_API_KEY'] = googleapi_key

    #from langchain.llms import Cohere
    #from langchain_community.llms import Cohere
    from langchain_google_genai import GoogleGenerativeAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.chains import SequentialChain
    from langchain.chains import ConversationChain
    #from langchain.memory import ConversationBufferWindowMemory
    import warnings

    warnings.filterwarnings('ignore')
    # health goal, med condition, fitness routine, preference

    #memory = ConversationBufferWindowMemory(k=10)

    # chain 1
    #llm = Cohere(temperature=0.7)
    llm = GoogleGenerativeAI(model="gemini-2.0-flash")

    prompt_template_name = PromptTemplate(
        input_variables =['health_goal'],
        template = " My health goal is {health_goal}. Please note and do not generate anything now."
    )

    hgoal_chain =LLMChain(llm=llm, prompt=prompt_template_name, output_key="hgoal")

    # chain 2
    #llm = Cohere(temperature=0.7)

    prompt_template_med_condition = PromptTemplate(
        input_variables = ['medical_conditions','hgoal'],
        template="My medical condition is {medical_conditions} and my health goal is {hgoal}.Please note and do not generate anything now."
    )

    med_cond_chain =LLMChain(llm=llm, prompt=prompt_template_med_condition, output_key="chain2")

    # chain 3
    #llm = Cohere(temperature=0.7)

    prompt_template_fitness_routine = PromptTemplate(
        input_variables = ['fitness_routines','chain2'],
        template="My fitness routine is {fitness_routines} and my previous data is {chain2}.Please note and do not generate anything now."
    )

    fitness_routines_chain =LLMChain(llm=llm, prompt=prompt_template_fitness_routine, output_key="chain3")


    # chain 4
    #llm = Cohere(temperature=0.7)

    prompt_template_preferences = PromptTemplate(
        input_variables = ['preferences','chain3'],
        template="My food preference is {preferences} and my previous data is {chain3}. Suggest a daily nutritional meal plan based on the given data. Explain why this meal plan is better."
    )

    preferences_chain =LLMChain(llm=llm, prompt=prompt_template_preferences, output_key="chain4")


    # sequential chain

    chain = SequentialChain(
        chains = [hgoal_chain, med_cond_chain, fitness_routines_chain, preferences_chain],
        #input_variables = ['health_goal'],
        input_variables = ['health_goal','medical_conditions','fitness_routines','preferences'],
        output_variables = ['hgoal','chain2','chain3','chain4']
    )

    response = chain.invoke({"health_goal": health_goal,"medical_conditions": medical_conditions,"fitness_routines": fitness_routines,"preferences":preferences})
    return response