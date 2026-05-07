class ExecutionContext:

    def __init__(self,
                 user_query):

        self.user_query = user_query

        self.agent_outputs = {}

        self.shared_memory = {}

    def add_agent_output(self,
                         agent_name,
                         output):

        self.agent_outputs[
            agent_name
        ] = output

    def get_agent_output(self,
                         agent_name):

        return self.agent_outputs.get(
            agent_name,
            None
        )

    def get_all_outputs(self):

        return self.agent_outputs
