# Bruker ParaVision scans -> one NIfTI series, with its b-values and b-vectors for diffusion scans.
#
# You need  a Bruker session folder and the scan numbers inside it; Julia 1.11.4 with env/julia (Fibers.jl).
# It does   reads the magnitude reconstruction (pdata/1) of every scan with Fibers.jl, in the order given, and joins their
#           volumes into one series. Two corrections follow, the same the paper's runs used:
#             - the image-to-world matrix: Fibers.jl (the pinned commit 29dc96a) builds it from visu_pars in the scanner's LPS
#               convention; it is turned into RAS, after reversing the slice axis when visu_pars says
#               disk_reverse_slice_order (the change the paper's conversions ran with, made to Fibers.jl on 7 Oct 2025). The
#               voxels are not moved;
#             - the b-values: rounded to the nearest 50 s/mm^2 (the scanner's effective values lie within a few s/mm^2 of the
#               nominal ones), and values up to 150 s/mm^2 set to 0.
#           On MF283 the image, b-values and b-vectors are byte-identical to the paper's.
#           Fibers.jl writes the image and, for diffusion scans, <out>.bvals and <out>.bvecs beside it.
# You get   <out>.nii.gz (+ <out>.bvals, <out>.bvecs)
#
#   julia --project=env/julia linc/construction/bruker_to_nifti.jl <session folder> <out>.nii.gz <scan> [<scan> ...]
#
# linc.construction.mri_raw runs it as its convert step.
using LinearAlgebra
# Plots' terminal-image extension (a dependency of Fibers.jl) fails to load with the package versions of record and prints an
# error; Fibers.jl does not use it. Its messages are kept off the terminal while Fibers.jl loads, as the lab's own scripts do.
redirect_stderr(devnull) do
    @eval using Fibers
end

"""true when the scan's visu_pars stores the slices in reverse order."""
function reversed_slices(scan)
    lines = readlines(joinpath(scan, "pdata", "1", "visu_pars"))
    i = findfirst(l -> startswith(l, "##\$VisuCoreDiskSliceOrder"), lines)
    return i !== nothing && strip(lines[i + 1]) == "disk_reverse_slice_order"
end

function main(args)
    length(args) >= 3 || error("usage: bruker_to_nifti.jl <session folder> <out>.nii.gz <scan> [<scan> ...]")
    scans = joinpath.(args[1], args[3:end])
    headers = mri_read.(scans; headeronly=true, reco=1)
    all(h -> h.image_type ∉ ("COMPLEX_IMAGE", "PHASE_IMAGE"), headers) || error("expected magnitude images")
    img = MRI(headers[1], sum(h.nframes for h in headers))
    v = 0
    for scan in scans
        one = mri_read(scan; reco=1)
        @views img.vol[:, :, :, v .+ (1:one.nframes)] .= one.vol
        v += one.nframes
    end

    lps = copy(img.vox2ras0)
    if reversed_slices(scans[1])
        lps[1:3, 3] .= -1 .* lps[1:3, 3]
        lps[1:3, 4] .= lps[1:3, 4] .+ (size(img.vol, 3) - 1) .* lps[1:3, 3]
    end
    img.vox2ras0 = Diagonal([-1, -1, 1, 1]) * lps

    bval = vcat(getfield.(headers, :bval)...)
    bval = round.(bval ./ 50) .* 50
    bval[bval .<= 150] .= 0
    img.bval = bval
    img.bvec = vcat(getfield.(headers, :bvec)...)
    mri_write(img, args[2])
end

main(ARGS)
