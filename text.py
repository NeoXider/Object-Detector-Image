import ast

def csharp_to_python(code):
    # СократитеCode
    tree = ast.parse(code)
    
    # Переставьте Code
    func_def = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            func_def = node
            break
    
    if func_def:
        new_code = []
        
        # Переименование функций
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                node.name = f"python_{func_def.name}"
        
        # Перекодирование Code
        for node in tree.body:
            if isinstance(node, ast.Assign):
                new_code.append(f"{node.target} = {ast.literal_eval(str(node.value))}")
            elif isinstance(node, ast.ListComp):
                new_code.append(f"def python_{{node.id}}: \n")
                for item in node.elts:
                    new_code.append(f"  python_{{item.id}} = python_{{item.name}}")
        for node in tree.body:
            if isinstance(node, ast.For):
                new_code.append("for i in python_{{node.iter}}:")
        
        # Объединить Code
        python_code = "\n".join(new_code)
        return python_code
    
    else:
        print("Сначала сгруппируйте функции!")
        return None

# Пример usage
csharp_code = """
public static void Main()
{
    int result = 0;
    foreach (var item in myList)
    {
        result += item;
    }
    Console.WriteLine(result);
}
"""

python_code = csharp_to_python(csharp_code)

if python_code:
    print(python_code)
else:
    print("ИспользованиеSharpPython неแนะนำ.")
