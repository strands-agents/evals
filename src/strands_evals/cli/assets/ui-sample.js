'use strict';

/* Shown when the page is opened from disk with no server behind it. */
const SAMPLE_EXPERIMENT = {
  "cases": [
    {
      "name": "order_status_lookup",
      "session_id": "10ee8de1-fd76-4bc6-b914-52c682a4fc75",
      "input": "Where is my order 11023?",
      "expected_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "expected_assertion": "The reply must state a status and include a tracking link.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      }
    },
    {
      "name": "password_reset",
      "session_id": "0f092995-5d61-447a-9d24-3aed47653cef",
      "input": "I can't log in and the reset email never arrives.",
      "expected_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "expected_assertion": "The reply must trigger a reset and name the address it was sent to.",
      "expected_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "auth"
        ],
        "turns": 1
      }
    },
    {
      "name": "refund_edge_case",
      "session_id": "1be21a8d-e97e-48e9-983f-a0e8226a1626",
      "input": "I was charged twice for order #88412 and the second charge is past the 30-day refund window. I want both refunded today.",
      "expected_output": "I refunded the duplicate charge of $42.00. The older charge is outside the 30-day window, so I escalated it to a supervisor for review.",
      "expected_assertion": "The reply must refund exactly one charge and escalate the second rather than promising a refund for it.",
      "expected_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "escalate_to_supervisor"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds",
          "policy-edge"
        ],
        "source": "session-2026-10-30-a91",
        "turns": 2
      }
    },
    {
      "name": "shipping_eta",
      "session_id": "17b8020e-e4b4-4d68-89dc-0cb5cf5575e8",
      "input": "When will order 77301 arrive?",
      "expected_output": "Order 77301 is scheduled to arrive Thursday. Weather in the region may delay it.",
      "expected_assertion": "The reply must give a date or date range.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      }
    },
    {
      "name": "cancel_order",
      "session_id": "50c71622-5131-4882-a077-d0ae612b3a03",
      "input": "Cancel order 55120 before it ships.",
      "expected_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "expected_assertion": "The reply must confirm the cancellation and name a refund timeline.",
      "expected_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "cancellation"
        ],
        "turns": 1
      }
    },
    {
      "name": "escalate_to_human",
      "session_id": "dc73c3b7-2886-4a2e-bb27-e389bea4c04a",
      "input": "This is the fourth time I've asked. Get me a person.",
      "expected_output": null,
      "expected_assertion": null,
      "expected_trajectory": null,
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "high",
        "tags": [
          "escalation"
        ],
        "turns": 1
      }
    },
    {
      "name": "duplicate_charge",
      "session_id": "ae714d8a-0876-4488-9d56-89839476952a",
      "input": "Two charges for the same order on my card.",
      "expected_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "expected_assertion": "The reply must refund exactly one charge.",
      "expected_trajectory": [
        "list_charges",
        "issue_refund"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds"
        ],
        "turns": 1
      }
    },
    {
      "name": "address_change",
      "session_id": "b7a9e6a3-181d-473f-a34e-5f4f26ebe3ee",
      "input": "I need to change the shipping address on order 90210.",
      "expected_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "expected_assertion": "The reply must either update the address or explain why it cannot.",
      "expected_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "logistics"
        ],
        "turns": 1
      }
    },
    {
      "name": "warranty_question",
      "session_id": "0be59053-419b-4ce1-9875-c66bcfa82964",
      "input": "Is the blender still under warranty? Bought it last March.",
      "expected_output": "The blender has a 1-year warranty and was bought in March, so it is still covered.",
      "expected_assertion": "The reply must state whether the warranty applies and why.",
      "expected_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "low",
        "tags": [
          "warranty"
        ],
        "turns": 1
      }
    },
    {
      "name": "return_then_refund",
      "session_id": "62286534-6da2-4b76-bd7a-9d8b25d6af83",
      "input": "I want to return the jacket. How long until I see the money? Can you expedite it?",
      "expected_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item, and we cannot expedite that.",
      "expected_assertion": "The reply must not promise an expedited refund.",
      "expected_trajectory": [
        "start_return",
        "get_refund_policy",
        "create_label"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 3
      }
    },
    {
      "name": "angry_customer_deesc",
      "session_id": "313a8dab-b000-42b2-8cdf-f5ffdbda10da",
      "input": "This is unacceptable. I want a manager NOW. Fine, just fix it.",
      "expected_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "expected_assertion": "The reply must not recite policy verbatim and must resolve the issue.",
      "expected_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "high",
        "tags": [
          "tone"
        ],
        "turns": 3
      }
    },
    {
      "name": "session_timeout",
      "session_id": "841815ef-1097-4ffd-89b5-14a9f0b38967",
      "input": "...are you still there? hello?",
      "expected_output": "I am still here. Your return on order 41900 is already started.",
      "expected_assertion": "The reply must confirm the pending action rather than restarting it.",
      "expected_trajectory": [
        "lookup_order",
        "get_return_status"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "low",
        "tags": [
          "reliability"
        ],
        "turns": 4
      }
    },
    {
      "name": "multi_item_dispute",
      "session_id": "60baaa50-cf5c-4648-aa2b-4fc1d44f45aa",
      "input": "Two of the four items are wrong. I only want to return those two.",
      "expected_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "expected_assertion": "The reply must return exactly two items.",
      "expected_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "expected_interactions": null,
      "expected_environment_state": null,
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 2
      }
    }
  ],
  "evaluators": [
    {
      "evaluator_type": "OutputEvaluator"
    },
    {
      "evaluator_type": "ToolTrajectoryEvaluator",
      "threshold": 0.9
    },
    {
      "evaluator_type": "RubricEvaluator",
      "name": "support_rubric"
    }
  ]
};

const SAMPLE_REPORT = {
  "overall_score": 0.9207407407407407,
  "scores": [
    1.0,
    1.0,
    0.53,
    0.71,
    1.0,
    0.0,
    1.0,
    1.0,
    0.96,
    0.86,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    0.75,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    0.67,
    1.0,
    0.0,
    1.0,
    0.6666666666666666,
    1.0,
    1.0,
    1.0,
    1.0,
    0.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0
  ],
  "cases": [
    {
      "input": "Where is my order 11023?",
      "actual_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "name": "order_status_lookup",
      "expected_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "expected_assertion": "The reply must state a status and include a tracking link.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "I can't log in and the reset email never arrives.",
      "actual_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "name": "password_reset",
      "expected_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "expected_assertion": "The reply must trigger a reset and name the address it was sent to.",
      "expected_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "actual_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "auth"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "I was charged twice for order #88412 and the second charge is past the 30-day refund window. I want both refunded today.",
      "actual_output": "I refunded the duplicate charge of $42.00 and I have also refunded the older charge of $42.00. Both should appear in 3-5 business days.",
      "name": "refund_edge_case",
      "expected_output": "I refunded the duplicate charge of $42.00. The older charge is outside the 30-day window, so I escalated it to a supervisor for review.",
      "expected_assertion": "The reply must refund exactly one charge and escalate the second rather than promising a refund for it.",
      "expected_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "escalate_to_supervisor"
      ],
      "actual_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds",
          "policy-edge"
        ],
        "source": "session-2026-10-30-a91",
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "When will order 77301 arrive?",
      "actual_output": "Order 77301 is scheduled to arrive Thursday.",
      "name": "shipping_eta",
      "expected_output": "Order 77301 is scheduled to arrive Thursday. Weather in the region may delay it.",
      "expected_assertion": "The reply must give a date or date range.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "Cancel order 55120 before it ships.",
      "actual_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "name": "cancel_order",
      "expected_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "expected_assertion": "The reply must confirm the cancellation and name a refund timeline.",
      "expected_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "actual_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "cancellation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "This is the fourth time I've asked. Get me a person.",
      "actual_output": "I am connecting you with a human agent now.",
      "name": "escalate_to_human",
      "expected_output": null,
      "expected_assertion": null,
      "expected_trajectory": null,
      "actual_trajectory": [
        "escalate_to_supervisor"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "escalation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "Two charges for the same order on my card.",
      "actual_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "name": "duplicate_charge",
      "expected_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "expected_assertion": "The reply must refund exactly one charge.",
      "expected_trajectory": [
        "list_charges",
        "issue_refund"
      ],
      "actual_trajectory": [
        "list_charges",
        "list_charges",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "I need to change the shipping address on order 90210.",
      "actual_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "name": "address_change",
      "expected_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "expected_assertion": "The reply must either update the address or explain why it cannot.",
      "expected_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "actual_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "logistics"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "Is the blender still under warranty? Bought it last March.",
      "actual_output": "The blender has a 1-year warranty and was bought in March, so it is likely still covered.",
      "name": "warranty_question",
      "expected_output": "The blender has a 1-year warranty and was bought in March, so it is still covered.",
      "expected_assertion": "The reply must state whether the warranty applies and why.",
      "expected_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "actual_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "metadata": {
        "priority": "low",
        "tags": [
          "warranty"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "I want to return the jacket. How long until I see the money? Can you expedite it?",
      "actual_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item. Expediting is not available.",
      "name": "return_then_refund",
      "expected_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item, and we cannot expedite that.",
      "expected_assertion": "The reply must not promise an expedited refund.",
      "expected_trajectory": [
        "start_return",
        "get_refund_policy",
        "create_label"
      ],
      "actual_trajectory": [
        "start_return",
        "create_label",
        "get_refund_policy"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "This is unacceptable. I want a manager NOW. Fine, just fix it.",
      "actual_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "name": "angry_customer_deesc",
      "expected_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "expected_assertion": "The reply must not recite policy verbatim and must resolve the issue.",
      "expected_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "actual_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "tone"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "...are you still there? hello?",
      "actual_output": "I am still here. Your return on order 41900 is already started.",
      "name": "session_timeout",
      "expected_output": "I am still here. Your return on order 41900 is already started.",
      "expected_assertion": "The reply must confirm the pending action rather than restarting it.",
      "expected_trajectory": [
        "lookup_order",
        "get_return_status"
      ],
      "actual_trajectory": [],
      "metadata": {
        "priority": "low",
        "tags": [
          "reliability"
        ],
        "turns": 4
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "Two of the four items are wrong. I only want to return those two.",
      "actual_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "name": "multi_item_dispute",
      "expected_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "expected_assertion": "The reply must return exactly two items.",
      "expected_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "actual_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "OutputEvaluator",
      "evaluator_type": "OutputEvaluator"
    },
    {
      "input": "Where is my order 11023?",
      "actual_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "name": "order_status_lookup",
      "expected_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "expected_assertion": "The reply must state a status and include a tracking link.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "I can't log in and the reset email never arrives.",
      "actual_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "name": "password_reset",
      "expected_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "expected_assertion": "The reply must trigger a reset and name the address it was sent to.",
      "expected_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "actual_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "auth"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "I was charged twice for order #88412 and the second charge is past the 30-day refund window. I want both refunded today.",
      "actual_output": "I refunded the duplicate charge of $42.00 and I have also refunded the older charge of $42.00. Both should appear in 3-5 business days.",
      "name": "refund_edge_case",
      "expected_output": "I refunded the duplicate charge of $42.00. The older charge is outside the 30-day window, so I escalated it to a supervisor for review.",
      "expected_assertion": "The reply must refund exactly one charge and escalate the second rather than promising a refund for it.",
      "expected_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "escalate_to_supervisor"
      ],
      "actual_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds",
          "policy-edge"
        ],
        "source": "session-2026-10-30-a91",
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "When will order 77301 arrive?",
      "actual_output": "Order 77301 is scheduled to arrive Thursday.",
      "name": "shipping_eta",
      "expected_output": "Order 77301 is scheduled to arrive Thursday. Weather in the region may delay it.",
      "expected_assertion": "The reply must give a date or date range.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "Cancel order 55120 before it ships.",
      "actual_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "name": "cancel_order",
      "expected_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "expected_assertion": "The reply must confirm the cancellation and name a refund timeline.",
      "expected_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "actual_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "cancellation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "This is the fourth time I've asked. Get me a person.",
      "actual_output": "I am connecting you with a human agent now.",
      "name": "escalate_to_human",
      "expected_output": null,
      "expected_assertion": null,
      "expected_trajectory": null,
      "actual_trajectory": [
        "escalate_to_supervisor"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "escalation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "Two charges for the same order on my card.",
      "actual_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "name": "duplicate_charge",
      "expected_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "expected_assertion": "The reply must refund exactly one charge.",
      "expected_trajectory": [
        "list_charges",
        "issue_refund"
      ],
      "actual_trajectory": [
        "list_charges",
        "list_charges",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "I need to change the shipping address on order 90210.",
      "actual_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "name": "address_change",
      "expected_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "expected_assertion": "The reply must either update the address or explain why it cannot.",
      "expected_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "actual_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "logistics"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "Is the blender still under warranty? Bought it last March.",
      "actual_output": "The blender has a 1-year warranty and was bought in March, so it is likely still covered.",
      "name": "warranty_question",
      "expected_output": "The blender has a 1-year warranty and was bought in March, so it is still covered.",
      "expected_assertion": "The reply must state whether the warranty applies and why.",
      "expected_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "actual_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "metadata": {
        "priority": "low",
        "tags": [
          "warranty"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "I want to return the jacket. How long until I see the money? Can you expedite it?",
      "actual_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item. Expediting is not available.",
      "name": "return_then_refund",
      "expected_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item, and we cannot expedite that.",
      "expected_assertion": "The reply must not promise an expedited refund.",
      "expected_trajectory": [
        "start_return",
        "get_refund_policy",
        "create_label"
      ],
      "actual_trajectory": [
        "start_return",
        "create_label",
        "get_refund_policy"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "This is unacceptable. I want a manager NOW. Fine, just fix it.",
      "actual_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "name": "angry_customer_deesc",
      "expected_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "expected_assertion": "The reply must not recite policy verbatim and must resolve the issue.",
      "expected_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "actual_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "tone"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "...are you still there? hello?",
      "actual_output": "I am still here. Your return on order 41900 is already started.",
      "name": "session_timeout",
      "expected_output": "I am still here. Your return on order 41900 is already started.",
      "expected_assertion": "The reply must confirm the pending action rather than restarting it.",
      "expected_trajectory": [
        "lookup_order",
        "get_return_status"
      ],
      "actual_trajectory": [],
      "metadata": {
        "priority": "low",
        "tags": [
          "reliability"
        ],
        "turns": 4
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "Two of the four items are wrong. I only want to return those two.",
      "actual_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "name": "multi_item_dispute",
      "expected_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "expected_assertion": "The reply must return exactly two items.",
      "expected_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "actual_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "ToolTrajectoryEvaluator",
      "evaluator_type": "ToolTrajectoryEvaluator"
    },
    {
      "input": "Where is my order 11023?",
      "actual_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "name": "order_status_lookup",
      "expected_output": "Order 11023 is in transit. Here is the carrier tracking link.",
      "expected_assertion": "The reply must state a status and include a tracking link.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "I can't log in and the reset email never arrives.",
      "actual_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "name": "password_reset",
      "expected_output": "I sent a reset link to the address on file. Check your spam folder if it does not arrive.",
      "expected_assertion": "The reply must trigger a reset and name the address it was sent to.",
      "expected_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "actual_trajectory": [
        "find_account",
        "send_password_reset"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "auth"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "I was charged twice for order #88412 and the second charge is past the 30-day refund window. I want both refunded today.",
      "actual_output": "I refunded the duplicate charge of $42.00 and I have also refunded the older charge of $42.00. Both should appear in 3-5 business days.",
      "name": "refund_edge_case",
      "expected_output": "I refunded the duplicate charge of $42.00. The older charge is outside the 30-day window, so I escalated it to a supervisor for review.",
      "expected_assertion": "The reply must refund exactly one charge and escalate the second rather than promising a refund for it.",
      "expected_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "escalate_to_supervisor"
      ],
      "actual_trajectory": [
        "lookup_order",
        "list_charges",
        "issue_refund",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds",
          "policy-edge"
        ],
        "source": "session-2026-10-30-a91",
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "When will order 77301 arrive?",
      "actual_output": "Order 77301 is scheduled to arrive Thursday.",
      "name": "shipping_eta",
      "expected_output": "Order 77301 is scheduled to arrive Thursday. Weather in the region may delay it.",
      "expected_assertion": "The reply must give a date or date range.",
      "expected_trajectory": [
        "lookup_order",
        "get_tracking"
      ],
      "actual_trajectory": [
        "lookup_order",
        "get_tracking",
        "get_tracking"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "status"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "Cancel order 55120 before it ships.",
      "actual_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "name": "cancel_order",
      "expected_output": "Order 55120 is cancelled. The refund posts in 3-5 business days.",
      "expected_assertion": "The reply must confirm the cancellation and name a refund timeline.",
      "expected_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "actual_trajectory": [
        "lookup_order",
        "cancel_order"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "cancellation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "This is the fourth time I've asked. Get me a person.",
      "actual_output": "I am connecting you with a human agent now.",
      "name": "escalate_to_human",
      "expected_output": null,
      "expected_assertion": null,
      "expected_trajectory": null,
      "actual_trajectory": [
        "escalate_to_supervisor"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "escalation"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "Two charges for the same order on my card.",
      "actual_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "name": "duplicate_charge",
      "expected_output": "I found the duplicate on order 66120 and refunded the later charge of $19.99.",
      "expected_assertion": "The reply must refund exactly one charge.",
      "expected_trajectory": [
        "list_charges",
        "issue_refund"
      ],
      "actual_trajectory": [
        "list_charges",
        "list_charges",
        "issue_refund"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "refunds"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "I need to change the shipping address on order 90210.",
      "actual_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "name": "address_change",
      "expected_output": "I updated the shipping address on order 90210. It has not shipped yet.",
      "expected_assertion": "The reply must either update the address or explain why it cannot.",
      "expected_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "actual_trajectory": [
        "lookup_order",
        "update_address"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "logistics"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "Is the blender still under warranty? Bought it last March.",
      "actual_output": "The blender has a 1-year warranty and was bought in March, so it is likely still covered.",
      "name": "warranty_question",
      "expected_output": "The blender has a 1-year warranty and was bought in March, so it is still covered.",
      "expected_assertion": "The reply must state whether the warranty applies and why.",
      "expected_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "actual_trajectory": [
        "lookup_purchase",
        "get_warranty_terms"
      ],
      "metadata": {
        "priority": "low",
        "tags": [
          "warranty"
        ],
        "turns": 1
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "I want to return the jacket. How long until I see the money? Can you expedite it?",
      "actual_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item. Expediting is not available.",
      "name": "return_then_refund",
      "expected_output": "I started the return and emailed a label. Refunds post 3-5 business days after we receive the item, and we cannot expedite that.",
      "expected_assertion": "The reply must not promise an expedited refund.",
      "expected_trajectory": [
        "start_return",
        "get_refund_policy",
        "create_label"
      ],
      "actual_trajectory": [
        "start_return",
        "create_label",
        "get_refund_policy"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "This is unacceptable. I want a manager NOW. Fine, just fix it.",
      "actual_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "name": "angry_customer_deesc",
      "expected_output": "You are right to be frustrated. I applied a $15.00 credit and the order is on its way.",
      "expected_assertion": "The reply must not recite policy verbatim and must resolve the issue.",
      "expected_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "actual_trajectory": [
        "lookup_order",
        "issue_credit"
      ],
      "metadata": {
        "priority": "high",
        "tags": [
          "tone"
        ],
        "turns": 3
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "...are you still there? hello?",
      "actual_output": "I am still here. Your return on order 41900 is already started.",
      "name": "session_timeout",
      "expected_output": "I am still here. Your return on order 41900 is already started.",
      "expected_assertion": "The reply must confirm the pending action rather than restarting it.",
      "expected_trajectory": [
        "lookup_order",
        "get_return_status"
      ],
      "actual_trajectory": [],
      "metadata": {
        "priority": "low",
        "tags": [
          "reliability"
        ],
        "turns": 4
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    },
    {
      "input": "Two of the four items are wrong. I only want to return those two.",
      "actual_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "name": "multi_item_dispute",
      "expected_output": "I started a return for the two incorrect items only. The other two stay on the order.",
      "expected_assertion": "The reply must return exactly two items.",
      "expected_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "actual_trajectory": [
        "lookup_order",
        "start_return",
        "create_label"
      ],
      "metadata": {
        "priority": "normal",
        "tags": [
          "returns"
        ],
        "turns": 2
      },
      "actual_interactions": null,
      "expected_interactions": null,
      "actual_environment_state": null,
      "expected_environment_state": null,
      "evaluator": "support_rubric",
      "evaluator_type": "RubricEvaluator"
    }
  ],
  "test_passes": [
    true,
    true,
    false,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    false,
    true,
    true,
    true,
    true,
    true,
    true,
    false,
    true,
    false,
    true,
    false,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true,
    true
  ],
  "reasons": [
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 0.53 vs expected_output (threshold 0.7)",
    "output similarity 0.71 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "no expected_output and no expected_assertion on the case, nothing to judge",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 0.96 vs expected_output (threshold 0.7)",
    "output similarity 0.86 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "output similarity 1.00 vs expected_output (threshold 0.7)",
    "matched 2 of 2 expected calls in order",
    "matched 2 of 2 expected calls in order",
    "matched 3 of 4 expected calls in order; missing escalate_to_supervisor",
    "matched 2 of 2 expected calls in order",
    "matched 2 of 2 expected calls in order",
    "no expected_trajectory on the case, nothing to compare",
    "matched 2 of 2 expected calls in order",
    "matched 2 of 2 expected calls in order",
    "matched 2 of 2 expected calls in order",
    "matched 2 of 3 expected calls in order",
    "matched 2 of 2 expected calls in order",
    "no trajectory provided",
    "matched 3 of 3 expected calls in order",
    "conciseness: 11 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 19 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 24 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 7 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 11 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "no expected_assertion on the case, so there is no rubric to apply",
    "conciseness: 14 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 13 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 17 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 22 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 18 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 12 words | no_policy_recitation: no boilerplate | actionable: names a concrete action",
    "conciseness: 17 words | no_policy_recitation: no boilerplate | actionable: names a concrete action"
  ],
  "detailed_results": [
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 0.53,
        "test_pass": false,
        "reason": "output similarity 0.53 vs expected_output (threshold 0.7)",
        "label": "incorrect",
        "metadata": {
          "threshold": 0.7,
          "similarity": 0.53
        }
      }
    ],
    [
      {
        "score": 0.71,
        "test_pass": true,
        "reason": "output similarity 0.71 vs expected_output (threshold 0.7)",
        "label": "partial",
        "metadata": {
          "threshold": 0.7,
          "similarity": 0.71
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 0.0,
        "test_pass": true,
        "reason": "no expected_output and no expected_assertion on the case, nothing to judge",
        "label": "not_applicable",
        "metadata": null
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 0.96,
        "test_pass": true,
        "reason": "output similarity 0.96 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 0.96
        }
      }
    ],
    [
      {
        "score": 0.86,
        "test_pass": true,
        "reason": "output similarity 0.86 vs expected_output (threshold 0.7)",
        "label": "partial",
        "metadata": {
          "threshold": 0.7,
          "similarity": 0.86
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "output similarity 1.00 vs expected_output (threshold 0.7)",
        "label": "correct",
        "metadata": {
          "threshold": 0.7,
          "similarity": 1.0
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "get_tracking"
          ],
          "actual": [
            "lookup_order",
            "get_tracking"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "find_account",
            "send_password_reset"
          ],
          "actual": [
            "find_account",
            "send_password_reset"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 0.75,
        "test_pass": false,
        "reason": "matched 3 of 4 expected calls in order; missing escalate_to_supervisor",
        "label": "missing_call",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "list_charges",
            "issue_refund",
            "escalate_to_supervisor"
          ],
          "actual": [
            "lookup_order",
            "list_charges",
            "issue_refund",
            "issue_refund"
          ],
          "missing": [
            "escalate_to_supervisor"
          ],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "get_tracking"
          ],
          "actual": [
            "lookup_order",
            "get_tracking",
            "get_tracking"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "cancel_order"
          ],
          "actual": [
            "lookup_order",
            "cancel_order"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 0.0,
        "test_pass": true,
        "reason": "no expected_trajectory on the case, nothing to compare",
        "label": "not_applicable",
        "metadata": null
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "list_charges",
            "issue_refund"
          ],
          "actual": [
            "list_charges",
            "list_charges",
            "issue_refund"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "update_address"
          ],
          "actual": [
            "lookup_order",
            "update_address"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_purchase",
            "get_warranty_terms"
          ],
          "actual": [
            "lookup_purchase",
            "get_warranty_terms"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 0.67,
        "test_pass": false,
        "reason": "matched 2 of 3 expected calls in order",
        "label": "extra_call",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "start_return",
            "get_refund_policy",
            "create_label"
          ],
          "actual": [
            "start_return",
            "create_label",
            "get_refund_policy"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 2 of 2 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "issue_credit"
          ],
          "actual": [
            "lookup_order",
            "issue_credit"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 0.0,
        "test_pass": false,
        "reason": "no trajectory provided",
        "label": "not_applicable",
        "metadata": null
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "matched 3 of 3 expected calls in order",
        "label": "match",
        "metadata": {
          "threshold": 0.9,
          "expected": [
            "lookup_order",
            "start_return",
            "create_label"
          ],
          "actual": [
            "lookup_order",
            "start_return",
            "create_label"
          ],
          "missing": [],
          "unexpected": []
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 11 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 0.0,
        "test_pass": false,
        "reason": "actionable: names a concrete action",
        "label": "unmet",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 19 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 24 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 7 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 11 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 0.0,
        "test_pass": true,
        "reason": "no expected_assertion on the case, so there is no rubric to apply",
        "label": "not_applicable",
        "metadata": null
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 14 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 13 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 17 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 22 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 18 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 12 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ],
    [
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "conciseness: 17 words",
        "label": "met",
        "metadata": {
          "criterion": "conciseness"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "no_policy_recitation: no boilerplate",
        "label": "met",
        "metadata": {
          "criterion": "no_policy_recitation"
        }
      },
      {
        "score": 1.0,
        "test_pass": true,
        "reason": "actionable: names a concrete action",
        "label": "met",
        "metadata": {
          "criterion": "actionable"
        }
      }
    ]
  ],
  "diagnoses": [
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null
  ],
  "recommendations": [
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null,
    null
  ]
};
