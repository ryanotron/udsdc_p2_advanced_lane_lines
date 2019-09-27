# Resubmission TODO
- [x] find, extract the difficult frames
- [x] try out the suggested colour spaces (LAB, LUV)
    - they are better than HLS (in cases that I've found, of course)
- [x] adjust ymid parameter for sans prior line fit
- [x] try out colour-only threshholding (no sobel!)
- [x] add discussion section to writeup
- [x] tidy up demo notebook
- [x] take a few more breaths
- [ ] resubmit

# P2 TODO

- [x] sanity checks
    - [x] lane width
    - [x] left-right curve similarity
        - |lcurve - rcurve| ~ lane_width
    - [x] left-right parallelism
        - check if a and b coefficients of a*y^2 + b*y + c are similar, but only for higher y (near camera)
- [x] insanity handling; if sanity checks fail, do what?
    - if two or more sanity checks fail, revoke has_prior status, try again once sans prior
- [x] try out canny edge
    - sobel threshholds are very low, maybe canny can have stronger distinction boundary?
    - result is not encouraging
- [x] spin out class to separate file
    - fix hsl --> hls while you're at that
- [x] retool notebook to use import that separate file
    - now, don't you regret not writing the class from the beginning?
- [x] collect pictures for writeup
- [x] do writeup
- [x] cleanup demo notebook
- [x] take a deep breath
- [x] submit project
