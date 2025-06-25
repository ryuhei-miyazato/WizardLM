base_instruction = (
    "あなたにはプロンプトリライターとして振る舞ってもらいます。\r\n"
    "目的は、有名なAIシステム（例：ChatGPTやGPT-4）が少し対応しづらくなるような、より複雑なバージョンに与えられたプロンプトを書き換えることです。\r\n"
    "ただし、書き換え後のプロンプトは合理的であり、人間が理解し応答できるものでなければなりません。\r\n"
    "また、#与えられたプロンプト#に含まれる表やコードなどの非テキスト部分、入力部分を省略してはいけません。\r\n"
    "次の方法を使ってプロンプトを複雑にしてください：\r\n"
    "{}\r\n\n"
    "ただし、書き換え後のプロンプト（#書き換え後のプロンプト#）は冗長にならないよう注意し、#与えられたプロンプト#に10～20語程度だけ追加してください。\r\n"
    "また、#与えられたプロンプト#、#書き換え後のプロンプト#、「与えられたプロンプト」、「作成されたプロンプト」という表現は#書き換え後のプロンプト#の中に登場させてはいけません。\r\n"
)

def createConstraintsPrompt(instruction):
    prompt = base_instruction.format("新たな制約や要件を1つ追加してください。")
    prompt += "#与えられたプロンプト#:\r\n{} \r\n".format(instruction)
    prompt += "#書き換え後のプロンプト#:\r\n"
    return prompt

def createDeepenPrompt(instruction):
    prompt = base_instruction.format("もし#与えられたプロンプト#が特定の事柄について尋ねている場合、その問いの深さや広がりを増してください。")
    prompt += "#与えられたプロンプト#:\r\n{} \r\n".format(instruction)
    prompt += "#書き換え後のプロンプト#:\r\n"
    return prompt

def createConcretizingPrompt(instruction):
    prompt = base_instruction.format("一般的な概念をより具体的な概念に置き換えてください。")
    prompt += "#与えられたプロンプト#:\r\n{} \r\n".format(instruction)
    prompt += "#書き換え後のプロンプト#:\r\n"
    return prompt

def createReasoningPrompt(instruction):
    prompt = base_instruction.format("もし#与えられたプロンプト#が簡単な思考過程で解ける場合は、複数ステップの推論を明示的に要求するように書き換えてください。")
    prompt += "#与えられたプロンプト#:\r\n{} \r\n".format(instruction)
    prompt += "#書き換え後のプロンプト#:\r\n"
    return prompt
