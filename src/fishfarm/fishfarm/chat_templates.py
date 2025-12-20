# This is the Japanese version of the famous Alpaca template. It is exactly the same as the one
# used in the evaluation of MGSM-JA in our evolutionary model merge paper.
ALPACA_JA = r"""
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{{ system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception(
            'Conversation roles must alternate user/assistant/user/assistant/...')}}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### 指示:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### 応答:\n' + message['content'].strip() + eos_token + '\n\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ '### 応答:' }}
    {% endif %}
{% endfor %}
""".replace(
    "    ", ""
).replace(
    "\n", ""
)


ALPACA_EN_COT = r"""
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}

{{ system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception(
            'Conversation roles must alternate user/assistant/user/assistant/...')}}
    {% endif %}

    {% if message['role'] == 'user' %}
        {{ '### Instruction:\n' + message['content'].strip() + '\n\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ '### Response:\n' + message['content'].strip() + eos_token + '\n\n' }}
    {% endif %}

    {% if loop.last and message['role'] == 'user' and add_generation_prompt %}
        {{ "### Response: Let's think step by step." }}
    {% endif %}
{% endfor %}
""".replace(
    "    ", ""
).replace(
    "\n", ""
)


LLAMA2 = """
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'] %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = false %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception(
            'Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 and system_message != false %}
        {% set content = '<<SYS>>\\n'
            + system_message
            + '\\n<</SYS>>\\n\\n'
            + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
""".replace(
    "    ", ""
).replace(
    "\n", ""
)

LLAMA3 = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>"
    "\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}"
    "{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)
