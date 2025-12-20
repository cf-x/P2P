# Instruction

Generate a targeted user profile for **{{ task_description }}** based on the provided user history data. This profile will be used to understand user behavior patterns specific to this task.

**IMPORTANT: You must analyze HOW this user behaves and makes decisions relevant to this task, NOT list WHAT specific content they interact with. Do not output lists of topics, keywords, or content examples. Focus only on behavioral tendencies, preferences, and decision-making patterns relevant to the target task.**

Focus on understanding patterns that inform task-specific user behavior:

1. **Task-Relevant User Preferences:**
   - Decision-making patterns and criteria relevant to the target task
   - Quality and content preferences that influence choices
   - Style and approach preferences in task-related activities
   - Consistency patterns in task-related decision making

2. **Behavioral Patterns Related to Task Performance:**
   - Interaction patterns and engagement styles relevant to the task
   - Response patterns to different types of content or options
   - Timing and frequency patterns in task-related activities
   - Adaptation patterns when encountering new or different scenarios

3. **Personal Style and Voice Indicators:**
   - Communication style patterns relevant to the task
   - Personal expression tendencies and voice characteristics
   - Authenticity markers and personal touch preferences
   - Consistency in personal style across different contexts

4. **Context Awareness and Adaptation Patterns:**
   - Awareness of audience, context, or requirements in task-related activities
   - Adaptation strategies for different scenarios within the task domain
   - Personalization approaches and individual preference integration
   - Balance between task requirements and personal style

# User History Data

{{ user_history }}

# Output Format

Output the user profile strictly in plain text describing the user's behavioral patterns, preferences, and decision-making tendencies specifically relevant to {{ task_description }}. Focus on patterns that predict how this user would approach and perform the target task.

**Do NOT output:**
- Lists of topics, keywords, or content they engage with
- Specific examples of content, products, or interactions
- Names of people, places, brands, or entities
- Content examples or subject matter details

**DO output:**
- Behavioral patterns and preferences relevant to the task
- Decision-making tendencies and criteria
- Personal style and approach patterns
- Task-specific interaction and engagement patterns

Derive insights strictly from the provided historical data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 