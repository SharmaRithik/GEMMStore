import re
import os

def extract_wgsl_kernels(file_path, output_dir):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Regular expression to extract WGSL sections
    wgsl_pattern = re.compile(r'// Dumped WGSL:(.*?)(?=(// Dumped WGSL:|/\* Dumped generated SPIRV disassembly \*/|$))', re.DOTALL)
    
    wgsl_kernels = [match[0] for match in wgsl_pattern.findall(content)]  # Extract first element from tuple
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each WGSL section to a separate file
    for i, kernel_code in enumerate(wgsl_kernels, start=1):
        filename = os.path.join(output_dir, f'kernel{i}.txt')
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(kernel_code.strip())
        print(f"Extracted: {filename}")

# Run the extraction
dump_file = "full_dump.txt"
output_directory = "/Users/rithik/GEMMStore/llm-models/deepseek_qwen-7B/fp16/shaders"
extract_wgsl_kernels(dump_file, output_directory)

