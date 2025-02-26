enable f16;

struct Matrix {
  size : vec2u,
  values : array<f16>,
}

@group(0) @binding(0) var<storage, read> matrixA : Matrix;
@group(0) @binding(1) var<storage, read> matrixB : Matrix;
@group(0) @binding(2) var<storage, read_write> resultMatrix : Matrix;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id : vec3u) {
  let row = global_id.x;
  let col = global_id.y;
  
  // Check bounds
  if (row >= matrixA.size.x || col >= matrixB.size.y) {
    return;
  }
  
  let width = matrixA.size.y;
  
  var sum : f16 = 0.0h;
  for (var i = 0u; i < width; i = i + 1u) {
    let a_idx = row * width + i;
    let b_idx = i * matrixB.size.y + col;
    sum = sum + (matrixA.values[a_idx] * matrixB.values[b_idx]);
  }
  
  let result_idx = row * matrixB.size.y + col;
  resultMatrix.values[result_idx] = sum;
}

