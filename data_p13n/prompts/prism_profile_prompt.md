# Instruction

Generate a targeted conversational user profile for **multi-turn dialogue participation** based on the provided conversation history and interactive communication data. This profile will be used to predict how this user would engage in natural conversations that reflect their personal communication style, interests, and personality traits across multiple dialogue turns.

**IMPORTANT: You must analyze HOW this user communicates and participates in conversations, NOT list WHAT topics they discuss. Do not output lists of conversation topics, subjects, or content keywords. Focus only on communication style patterns and conversational behavioral tendencies.**

Focus on understanding conversation adaptation patterns:

1. **Multi-Turn Conversation Style and Dialogue Flow:**
   - Characteristic response patterns and turn-taking behaviors
   - Adaptation of communication style across multiple conversation turns
   - Consistency in personality and voice throughout extended dialogues
   - Response length and detail preferences in different conversation contexts

2. **Conversational Topic Engagement and Interest Expression:**
   - Natural topic introduction and conversation steering patterns
   - Personal interest integration and authentic engagement in dialogues
   - Handling of topic transitions and conversation flow management
   - Expression of genuine opinions and perspectives in multi-turn contexts

3. **Interactive Communication Behavioral Patterns:**
   - Collaborative vs. directive conversation participation styles
   - Question-asking vs. information-sharing balance in dialogues
   - Empathy expression and emotional intelligence in extended conversations
   - Adaptation to conversation partners and context-appropriate responses

4. **Dialogue Personality and Authenticity Indicators:**
   - Personal communication traits that emerge in natural conversation
   - Authentic interest patterns and genuine engagement markers
   - Social bonding and relationship building through conversation
   - Privacy boundaries and personal information sharing in dialogue contexts


# User History Data

{{ user_history }}

# Output Format

Output the conversational user profile strictly in plain text describing the user's conversation patterns and communication behavior tendencies. Focus specifically on patterns that predict **how this user would naturally participate in multi-turn conversations** reflecting their authentic communication style and personality.

**Do NOT output:**
- Lists of conversation topics or subjects
- Specific discussion content or keywords
- Names of people, places, or topics discussed
- Conversation content examples

**DO output:**
- Conversational style patterns and preferences
- Communication behavioral patterns
- Dialogue participation tendencies
- Interactive communication personality markers

Derive insights strictly from the provided conversation history and interaction data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 