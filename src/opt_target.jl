abstract type AbstractOpt end

struct Opt <: AbstractOpt
    opt_target::Symbol
end

struct ControlOpt<:Opt end
struct StateOpt<:Opt  end
struct MeasurementOpt<:Opt end
struct CompOpt<:Opt end
struct StateControlOpt<:CompOpt end
struct StateMeasurementOpt<:CompOpt end
struct StateControlMeasurementOpt<:CompOpt end

ControlOpt() = ControlOpt(:Copt)
StateOpt() = StateOpt(:Sopt)
MeasurementOpt() = MeasurementOpt(:Mopt)
StateControlOpt() = StateControlOpt(:SCopt)
StateMeasurementOpt() = StateMeasurementOpt(:SMopt)
StateControlMeasurementOpt() = StateControlMeasurementOpt(:SCMOpt)