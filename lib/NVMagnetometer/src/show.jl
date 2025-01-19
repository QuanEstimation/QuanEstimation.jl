function Base.show(io::IO, ::MIME"text/plain", t::NVMagnetometerScheme)
    @unpack data, io_hooks = t
    @unpack D, gS, gI, A1, A2, B1, B2, B3, γ, init_state, ctrl, tspan, M = data
    print("""
          NVMagnetometerScheme 
          ├─ StatePreparation => DensityMatrix
          │  ├─ ndim = $(size(init_state))
          │  └─ ψ0   = $(init_state)
          ├─ Parameterization => LindbladDynamics
          │  ├─ tspan = $(tspan)    
          │  ├─ Hamiltonian => NVCenterHamiltonian
          │  │  ├─ D  = $(D)   
          │  │  ├─ gS = $(gS)
          │  │  ├─ gI = $(gI)
          │  │  ├─ A1 = $(A1)
          │  │  ├─ A2 = $(A2)
          │  │  └─ B  = [$(B1), $(B2), $(B3)]
          │  ├─ Controls
          │  │  ├─ Hc = [S1, S2, S3]
          │  │  └─ ctrl = $(ctrl)
          │  ├─ decays
          │  │  ├─ decay_opt => [S3]
          │  │  └─ γ  = $(γ)
          └─ Measurement
              └─ M = $(M)

          """)
end
