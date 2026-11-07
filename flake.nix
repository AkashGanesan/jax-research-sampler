{
  description = "ai-overview: ML research study env (nix = tools, uv = python + packages)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
    in {
      devShells.${system}.default = pkgs.mkShellNoCC {  # no nix gcc: system nvcc/triton need the system compiler
        # Deliberately NO nix python / cudaPackages: on non-NixOS they clash with
        # the system NVIDIA driver (libcuda.so). uv manages python; CUDA comes from wheels.
        packages = with pkgs; [ uv git nvtopPackages.nvidia ruff ];

        shellHook = ''
          export UV_PYTHON_PREFERENCE=only-managed
          unset PYTHONPATH LD_LIBRARY_PATH
          # JAX: don't grab the whole GPU by default (override for benchmarks / long runs).
          export XLA_PYTHON_CLIENT_MEM_FRACTION=''${XLA_PYTHON_CLIENT_MEM_FRACTION:-.75}
          export JAX_COMPILATION_CACHE_DIR="$PWD/.cache/jax"
          # vLLM: use the built-in sampler; FlashInfer's sampler JIT needs nvcc + gcc<=14.
          export VLLM_USE_FLASHINFER_SAMPLER=0
        '';
      };
    };
}
