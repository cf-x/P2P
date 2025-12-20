# Instruction

Generate a targeted headline writing user profile for **news article headline generation** based on the provided news reading and writing history data. This profile will be used to predict what style of headlines this user would create for news articles.

**IMPORTANT: You must analyze HOW this user writes headlines, NOT list WHAT news topics they read. Do not output lists of news topics, events, or keywords. Focus only on headline writing style patterns and behavioral tendencies.**

Focus on understanding headline creation patterns:

1. **Headline Creation Style and Structure Preferences:**
   - Preferred headline length and word choice patterns
   - Use of engaging vs. straightforward informational headlines
   - Balance between attention-grabbing elements and factual accuracy
   - Tone and voice consistency in headline writing (formal, casual, dramatic, neutral)

2. **News Story Framing and Summarization Approach:**
   - Emphasis on key story elements in headline construction (who, what, when, where, why)
   - Approach to capturing complex news stories in concise headlines
   - Handling of breaking news urgency vs. feature story depth
   - Use of action verbs vs. descriptive language in headlines

3. **Headline Writing Behavioral Patterns:**
   - Systematic vs. creative approach to headline generation
   - Adaptation of headline style for different news categories and topics
   - Incorporation of trending language and current event terminology
   - Consistency in headline quality and engagement across different article types

4. **Editorial Voice and Communication Indicators:**
   - Personal writing voice that emerges in headline creation
   - Audience awareness and target readership consideration
   - Use of persuasive language and emotional appeals in headlines
   - Journalistic integrity vs. engagement optimization in headline choices


# User History Data

{{ user_history }}

# Output Format

Output the news headline writing user profile strictly in plain text describing the user's headline creation patterns and writing style tendencies. Focus specifically on patterns that predict **what style of headlines this user would generate for news articles** based on their reading preferences and communication patterns.

**Do NOT output:**
- Lists of news topics or events
- Current events or news keywords
- Political topics or news categories
- Specific news content examples

**DO output:**
- Headline writing style patterns and preferences
- News framing and structure tendencies
- Editorial voice and communication behavioral patterns
- Headline creation approach and methodology

Derive insights strictly from the provided news interaction and reading data. Do not include explanations, introductions, headings, bullet points, or any formatting structure. 