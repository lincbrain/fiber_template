# SANDI on the microstructure diffusion session, as for the paper's data (Microstructure.jl).
#
# You need  the session's bias-corrected series <data>.nii.gz with <data>.bvals and <data>.bvecs beside it, its brain mask,
#           the per-volume timing files <times>.techo, <times>.tdelta, <times>.tsmalldel (ms); Julia 1.11.4 with env/julia.
# It does   1. the spherical mean: the signal averaged over the directions of each b-shell and divided by the b0, and the b0's
#              signal-to-noise ratio (Microstructure.jl spherical_mean, which writes diravg*.nii.gz and snr_b0.nii.gz beside
#              <data>);
#           2. two models, each fitted by a small neural network trained on 50 000 signals simulated at the data's noise level
#              (three hidden layers of 32 units, dropout 0.1, 100 epochs on the CPU, random seed 1) and then applied to every
#              voxel of the mask 100 times with dropout on; the mean and the standard deviation of the 100 are the estimate and
#              its uncertainty. First SANDI (soma, neurite, extracellular space), then SANDIdot (SANDI with a dot compartment).
#              The MRI build reads SANDIdot's compartment fractions. SANDI runs first because SANDIdot's network starts from
#              the random state SANDI's run leaves behind, as in the paper's run: with this order SANDIdot's estimates are
#              identical to the paper's, while SANDI's own estimates change from run to run (its network starts unseeded).
# You get   <out>/SANDI/ and <out>/SANDIdot/: est.<parameter>.mean.nii.gz and .std.nii.gz (fracs: soma, neurite[, dot];
#           soma.size; extra.diff) and sigma_for_training.txt (the noise range the networks were trained on)
#
#   julia --project=env/julia linc/construction/sandi.jl <data>.nii.gz <brain mask>.nii.gz <times> <out folder>
#
# linc.construction.mri_raw runs it as the last step of its dt2 command.
#
# On MF283, with the paper's inputs, the spherical means and every SANDIdot output are byte-identical to the paper's.
using Distributions, Random, DelimitedFiles, Statistics
using Flux: relu6, cpu
redirect_stderr(devnull) do          # Plots' terminal-image extension prints a harmless error while Fibers.jl loads
    @eval using Microstructure, Fibers
end

function fit(model, prior_dist, Sl, snr, protocol, mask, savedir)
    mkpath(savedir)
    sigma_range, sigma_dist = sigma_level(snr, mask)
    writedlm(joinpath(savedir, "sigma_for_training.txt"), (sigma_range..., sigma_dist.μ, sigma_dist.σ))
    params = ("soma.size", "extra.diff", "fracs")
    prior_range = ((1.0e-6, 10.0e-6), (0.2e-9, 1.2e-9), (0.0, 1.0))
    netarg = NetworkArg(model, protocol, params, prior_range, prior_dist, (), "Gaussian", sigma_range, sigma_dist, 50000,
                        length(protocol.bval), length(params) + length(model.fracs) - 1, (32, 32, 32), (0.1, 0.1, 0.1), relu6)
    trainarg = TrainingArg(batchsize=128, lossf=losses_rmse, device=cpu)
    mlp, logs, inputs, labels = training(trainarg, netarg)
    test(mlp, inputs, 100)           # the paper's run evaluated the network here; kept, as it advances the random state
    nn_estimate(Sl, mask, mlp, netarg, 100, savedir, "est.")
end

function main(args)
    length(args) == 4 || error("usage: sandi.jl <data>.nii.gz <brain mask>.nii.gz <times> <out folder>")
    data, maskfile, times, out = args
    stem = replace(data, r"\.nii(\.gz)?$" => "")
    Sl, protocol, snr_b0 = spherical_mean(data, true, (stem .* [".bvals", ".bvecs"])..., (times .* [".techo", ".tdelta", ".tsmalldel"])...)
    snr = MRI(Sl, 2); snr.vol = snr_b0; snr.nframes = 1
    mask = mri_read(maskfile)
    fit(SANDI(soma=Microstructure.Sphere(diff=2.0e-9, size=5.0e-9), neurite=Stick(dpara=0.6e-9), extra=Iso(diff=0.6e-9),
              fracs=[0.4, 0.4]), (nothing, nothing, Dirichlet([1, 1, 0.3])), Sl, snr, protocol, mask, joinpath(out, "SANDI"))
    fit(SANDIdot(soma=Microstructure.Sphere(diff=2.0e-9, size=5.0e-6), neurite=Stick(dpara=0.6e-9), extra=Iso(diff=0.6e-9),
                 fracs=[0.4, 0.4, 0.1]), (nothing, nothing, Dirichlet([1, 1, 0.3, 0.3])), Sl, snr, protocol, mask,
        joinpath(out, "SANDIdot"))
end

main(ARGS)
