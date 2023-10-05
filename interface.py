import gradio as gr
import functions
import build_knowledge


with gr.Blocks() as pre_ui:
    with gr.Column():
        gr.Markdown(
            """
            # Build Knowledge Base
            """
        )
        with gr.Row():
            smart_access_token = gr.Textbox(label="SmartSheet Access Token")
            sheet_id = gr.Textbox(label="Sheet ID")
        with gr.Row():
            _opeanai_key = gr.Textbox(show_label=False, placeholder="OpenAI API Key")
            _model_name = gr.Dropdown(
                [
                    "gpt-4-0314",
                    "gpt-4-0613",
                    "gpt-4",
                    "gpt-3.5-turbo-0613",
                    "gpt-3.5-turbo-0301",
                    "gpt-3.5-turbo-16k-0613",
                ],
                show_label=False,
            )
        with gr.Row():
            result = gr.Textbox(label="Result")
            b_build = gr.Button("Build")
    
    def build_knowledge_base(access_token, sheet_id, openai_key, model_name):
        phases = build_knowledge.build_knowledge_base(access_token, sheet_id, openai_key, model_name)
        _phases = phases[2: ] if phases[1] == 'baseline' else phases[1:]
        return 'Success!', gr.Dropdown(_phases), gr.Dropdown(phases)


with gr.Blocks() as run_ui:
    gr.Markdown(
        """
        # Project Schedule Analysis
        """
    )
    
    with gr.Row():
        opeanai_key = gr.Textbox(show_label=False, placeholder="OpenAI API Key")
        model_name = gr.Dropdown(
            [
                "gpt-4-0314",
                "gpt-4-0613",
                "gpt-4",
                "gpt-3.5-turbo-0613",
                "gpt-3.5-turbo-0301",
                "gpt-3.5-turbo-16k-0613",
            ],
            show_label=False,
        )

        
    with gr.Row():
        with gr.Column():
            analysis_text = gr.TextArea(label="Detailed Analysis", lines=15, max_lines=15)
            with gr.Row():
                choose_phase = gr.Dropdown([])
                b_phase = gr.Button("Get Analysis")
        with gr.Column():
            choice = gr.Dropdown(
                [],
                label="Pinecone Index",
            )
            query = gr.Textbox(label="Query:")
            banswer = gr.Button("Generate Answer")
            answer = gr.TextArea(label="Answer:", lines=6, max_lines=10)


    def set_analysis(opeanai_key, model_name, phase_text):
        response = functions.generate_analysis(
            opeanai_key, model_name, phase_text
        )
        return response

    def set_answer(opeanai_key, model_name, index_name, query):
        if index_name == "" or query == "":
            return "Choose Index and Input QUERY!!!"
        response = functions.generate_answer(
            opeanai_key, model_name, index_name, query
        )
        return response

    b_build.click(build_knowledge_base, inputs=[smart_access_token, sheet_id, _opeanai_key, _model_name], outputs=[result, choose_phase, choice])
        
    b_phase.click(set_analysis, [opeanai_key, model_name, choose_phase], outputs=analysis_text)
    banswer.click(set_answer, [opeanai_key, model_name, choice, query], outputs=answer)
    

demo = gr.TabbedInterface([pre_ui, run_ui], ["Build Knowledge-Base", "Run System"])
demo.queue().launch(share=True)
