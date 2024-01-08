from create_chain import Chain
from langchain.chat_models import ChatOpenAI
import time

gptJsonModel = ChatOpenAI(
    #models : https://platform.openai.com/docs/models/gpt-3-5
    #model="gpt-4-1106-preview",
    model="gpt-3.5-turbo-1106",
    model_kwargs={
        "response_format": {
            "type": "json_object"
        }
    }
)

def main():

    # Define input
    # TODO: take input from API
    input = {
            "type": "bodyweight",
            "gender": "male",
            "level": "intermediate",
            "frequency": "4",
            "goal": "building muscles",
            "size": "175",
            "weight": "75",
            "age": "26"
        }
    
    # Record the start time
    start_time = time.time()

    # Provide evidences prompt template
    # TODO: Add prompt to Hub (https://docs.smith.langchain.com/cookbook/hub-examples)
    PE_system_template = """You are a worldwide known scientist specialized in body science and fitness training.
    Your goal is to provide science-based evidences for a customized fitness training program tailored to the individual's goal of {goal}. 
    Your recommendations should be backed by scientific research in the field of body science and should consider physiological data."""
    PE_human_template="""Please provide detailed and informative science-based evidences to assist in creating a tailored fitness plan for a person is a {gender}, {size}cm tall, weights {weight}.
    The evidences should be related to any factor that influence the fitness plan to achieve the goal of {goal}, such as training method, intensity, number of reps, number of sets, type of exercices, techniques and rest.
    Return the evidences as a JSON and feel free to quote meta-analysis, studies, articles.
    The JSON should be structured as follow: "evidences": [ "topic": "training", "title": "title", "description": "description", "references": "references" ]"""
    
    # Create PE Chain
    PEChain = Chain(PE_system_template, PE_human_template, gptJsonModel)
    
    # Invoke
    PE_output_dict = PEChain.invoke(**input)

    # Concatenate dicts
    GM_input = {**input, **PE_output_dict}

    # Generate methods prompt template
    # TODO: Add prompt to Hub (https://docs.smith.langchain.com/cookbook/hub-examples)
    GM_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven fitness programs rooted in science.
    Based on the science-based evidences that the user will pass you, you will generate the best fitness {type} training methods for a {age} years old {gender} person. 
    Explain briefly how those methods are related with the provided scientific evidences and why would they fit perfectly to this person.
    Return the training method as a JSON that follows this structure: "methods": [ "name": "name", "description": "description", "execution": "execution", "reference_to_evidence": "reference_to_evidence", "tailored": "tailored" ]"""
    GM_human_template="""{evidences}"""

    # Create GM Chain
    GMChain = Chain(GM_system_template, GM_human_template, gptJsonModel)

    # Invoke
    GM_output_dict = GMChain.invoke(**GM_input)

    # Concatenate dicts
    WP_input = {**input, **GM_output_dict}

    ## Generate weekly program prompt template
    # TODO: Add prompt to Hub (https://docs.smith.langchain.com/cookbook/hub-examples)
    WP_system_template = """You are a top-tier personal trainer known for crafting unique, result-driven programs rooted in science.
    Based on the training methods the user will send, design a distinguished four-week {type} training schedule for a {gender} at an {level} level with a {level} level. 
    Each week MUST have exactly {frequency} sessions.
    You MUST return the program formatted as JSON object with the following fields: weeks [ weekNumber, weekDescription, sessions [ sessionNumber, goal, description, reference_to_method, exercises [ name, description, sets, reps, restTime ] ] ]."""
    WP_user_template = """{methods}"""

    # Create WP Chain
    WPChain = Chain(WP_system_template, WP_user_template, gptJsonModel)

    # Invoke
    WP_output_dict = WPChain.invoke(**WP_input)

    print(str(WP_output_dict))

    # Record the end time
    end_time = time.time()

    duration = round(end_time - start_time, 2)
    print(f"Time : {duration} seconds")

    assert len(WP_output_dict['weeks']) == 4, "The number of weeks is not correct"

    for week in WP_output_dict['weeks']:
        assert len(week['sessions']) == int(WP_input['frequency']), "The number of sessions per week is not correct"

    # TODO: Add program reviews

if __name__ == "__main__":
    main()