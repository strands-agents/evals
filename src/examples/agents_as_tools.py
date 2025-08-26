import asyncio
import datetime

from strands import Agent, tool
from strands_evals import Case, Dataset
from strands_evals.evaluators import InteractionsEvaluator, TrajectoryEvaluator
from strands_evals.extractors import tools_use_extractor
from strands_evals.types import EvaluationReport, Interaction


async def async_agents_as_tools_example() -> tuple[EvaluationReport, EvaluationReport]:
    """
    Demonstrates customer support orchestrator using specialized agents as tools.

    This example:
    1. Defines a task function with an orchestrator that routes to specialized support agents
    2. Creates test cases for different customer support scenarios
    3. Creates TrajectoryEvaluator and InteractionsEvaluator to assess routing decisions
    4. Creates datasets with the test cases and evaluators
    5. Runs evaluations and analyzes the reports

    Returns:
        tuple[EvaluationReport, EvaluationReport]: The trajectory and interaction evaluation results
    """

    ### Step 1: Define task ###
    def customer_support(task: str) -> dict:
        # Fake database for customer support
        FAKE_DATABASE: dict[str, dict | list] = {
            "customers": {
                "user123": {
                    "name": "John Doe",
                    "email": "john@example.com",
                    "subscription": "Pro",
                    "last_payment": "2024-01-15",
                },
                "user456": {
                    "name": "Jane Smith",
                    "email": "jane@example.com",
                    "subscription": "Basic",
                    "last_payment": "2024-01-10",
                },
            },
            "products": {
                "pro_plan": {
                    "name": "Pro Plan",
                    "price": 29.99,
                    "features": ["Advanced analytics", "Priority support", "Custom integrations"],
                },
                "basic_plan": {"name": "Basic Plan", "price": 9.99, "features": ["Basic analytics", "Email support"]},
            },
            "orders": {
                "order789": {
                    "customer_id": "user123",
                    "product": "pro_plan",
                    "status": "shipped",
                    "date": "2024-01-20",
                },
                "order101": {
                    "customer_id": "user456",
                    "product": "basic_plan",
                    "status": "processing",
                    "date": "2024-01-22",
                },
            },
            "tickets": [],
        }

        @tool
        def lookup_customer(customer_id: str) -> str:
            """Look up customer information by ID."""
            customer = (
                FAKE_DATABASE["customers"].get(customer_id) if isinstance(FAKE_DATABASE["customers"], dict) else None
            )
            if customer:
                return (
                    f"Customer {customer_id}: {customer['name']} ({customer['email']} "
                    f"- {customer['subscription']} plan, last payment: {customer['last_payment']}"
                )
            return f"Customer {customer_id} not found"

        @tool
        def lookup_product(product_id: str) -> str:
            """Look up product information by ID."""
            product = FAKE_DATABASE["products"].get(product_id) if isinstance(FAKE_DATABASE["products"], dict) else None
            if product:
                features = ", ".join(product["features"])
                return f"Product {product_id}: {product['name']} - ${product['price']}/month. Features: {features}"
            return f"Product {product_id} not found"

        @tool
        def lookup_order(order_id: str) -> str:
            """Look up order information by ID."""
            order = FAKE_DATABASE["orders"].get(order_id) if isinstance(FAKE_DATABASE["orders"], dict) else None
            if order:
                return (
                    f"Order {order_id}: Customer {order['customer_id']},"
                    f" Product {order['product']}, Status: {order['status']}, Date: {order['date']}"
                )
            return f"Order {order_id} not found"

        @tool
        def create_ticket(customer_id: str, issue_type: str, description: str) -> str:
            """Create a support ticket."""
            import datetime

            ticket_id = f"ticket{len(FAKE_DATABASE['tickets']) + 1}"
            ticket = {
                "id": ticket_id,
                "customer_id": customer_id,
                "issue_type": issue_type,
                "description": description,
                "status": "open",
                "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            if isinstance(FAKE_DATABASE["tickets"], list):
                FAKE_DATABASE["tickets"].append(ticket)
            return f"Created {ticket_id} for customer {customer_id}: {issue_type} - {description}"

        # agents tools
        @tool
        def technical_support(query: str) -> str:
            """Handle technical issues, bugs, and troubleshooting."""
            agent = Agent(
                system_prompt=(
                    "You are a technical support specialist. Help users troubleshoot technical issues, bugs,"
                    " and product functionality problems. Use available tools to look up customer"
                    " info and create tickets."
                ),
                tools=[lookup_customer, create_ticket],
                callback_handler=None,
            )
            return str(agent(query))

        @tool
        def billing_support(query: str) -> str:
            """Handle billing, payments, and account issues."""
            agent = Agent(
                system_prompt=(
                    "You are a billing specialist. Help with payment issues, subscription questions, "
                    "refunds, and account billing problems. Use tools to look up customer information."
                ),
                tools=[lookup_customer],
                callback_handler=None,
            )
            return str(agent(query))

        @tool
        def product_info(query: str) -> str:
            """Provide product information and feature explanations."""
            agent = Agent(
                system_prompt=(
                    "You are a product specialist. Explain product features, capabilities, and help users"
                    " understand how to use products effectively. Use tools to look up product details."
                ),
                tools=[lookup_product],
                callback_handler=None,
            )
            return str(agent(query))

        @tool
        def returns_exchanges(query: str) -> str:
            """Handle returns, exchanges, and any order issues."""
            agent = Agent(
                system_prompt=(
                    "You are a returns specialist. Help with product returns, exchanges, order "
                    "modifications, and shipping issues. Use tools to look up order and customer information."
                ),
                tools=[lookup_order, lookup_customer],
                callback_handler=None,
            )
            return str(agent(query))

        # Customer support orchestrator
        CUSTOMER_SUPPORT_PROMPT = """
        You are a customer support router. Direct queries to the appropriate specialist:
        - Technical issues, bugs, errors → technical_support
        - Billing, payments, subscriptions → billing_support  
        - Product questions, features → product_info
        - Returns, exchanges, orders → returns_exchanges
        - Simple greetings → answer directly
        """

        orchestrator = Agent(
            system_prompt=CUSTOMER_SUPPORT_PROMPT,
            callback_handler=None,
            tools=[technical_support, billing_support, product_info, returns_exchanges],
        )
        response = orchestrator(task)
        description = tools_use_extractor.extract_tools_description(orchestrator)
        trajectory_evaluator.update_trajectory_description(description)
        interaction_evaluator.update_interaction_description(description)
        tools_used = tools_use_extractor.extract_agent_tools_used_from_messages(orchestrator.messages)

        interactions = []
        for tool_used in tools_used:
            interactions.append(
                Interaction(
                    node_name=tool_used["name"], dependencies=[tool_used["input"]], messages=[tool_used["tool_result"]]
                )
            )
        return {"output": str(response), "trajectory": tools_used, "interactions": interactions}

    ### Step 2: Create test cases ###
    test_cases = [
        Case(input="My app keeps crashing when I try to login for user123"),
        Case(input="I was charged twice this month, my customer ID is user456"),
        Case(input="What features are included in the pro_plan?"),
        Case(input="I want to check the status of order789"),
    ]

    ### Step 3: Create evaluator ###
    trajectory_evaluator = TrajectoryEvaluator(
        rubric=(
            "Does the orchestrator route customer queries to the appropriate support specialist? "
            "Score 1.0 if the routing is correct (technical→technical_support, billing→billing_support, "
            "product→product_info, orders→returns_exchanges), 0.5 if partially correct, 0.0 if incorrect."
        )
    )
    interaction_evaluator = InteractionsEvaluator(
        rubric=(
            "Does each specialist agent use the appropriate tools and provide complete, helpful responses? "
            "Score 1.0 if tools are used correctly and response is comprehensive, 0.5 if partially adequate, "
            "0.0 if inadequate."
        )
    )

    ### Step 4: Create dataset ###
    dataset_trajectory = Dataset(cases=test_cases, evaluator=trajectory_evaluator)
    dataset_interactions = Dataset(cases=test_cases, evaluator=interaction_evaluator)

    ### Step 5: Run the evaluation ###
    report_trajectory = await dataset_trajectory.run_evaluations_async(customer_support)
    report_interactions = await dataset_interactions.run_evaluations_async(customer_support)

    return report_trajectory, report_interactions


if __name__ == "__main__":
    # run the file as a module: eg. python -m examples.agents_as_tools
    start = datetime.datetime.now()
    report_trajectory, report_interactions = asyncio.run(async_agents_as_tools_example())
    end = datetime.datetime.now()

    report_trajectory.to_file("customer_support_orchestrator_report_trajectory", "json")
    report_trajectory.display(include_actual_trajectory=True)

    report_interactions.to_file("customer_support_orchestrator_report_interactions")
    report_interactions.display(include_actual_interactions=True)
