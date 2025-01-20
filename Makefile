JL = julia --project

default: init test

init:
	$(JL) -e 'using Pkg; Pkg.develop("QuanEstimation"); Pkg.develop([Pkg.PackageSpec(path = joinpath("lib", pkg)) for pkg in ["QuanEstimationBase", "NVMagnetometer"]]); Pkg.precompile()'

update:
	$(JL) -e 'using Pkg; Pkg.update(); Pkg.precompile()'

update-docs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); Pkg.update(); Pkg.precompile()'

test:
	$(JL) -e 'using Pkg; Pkg.test("QuanEstimation")'

test-%:
	$(JL) -e 'using Pkg; Pkg.test("$*")'

coverage:
	$(JL) -e 'using Pkg; Pkg.test(["QuanEstimation", "QuanEstimationBase", "NVMagnetometer"]; coverage=true)'

servedocs:
	$(JL) -e 'using Pkg; Pkg.activate("docs"); using LiveServer; servedocs(;skip_dirs=["docs/src/assets", "docs/src/generated"])'

clean:
	rm -rf docs/build
	find . -name "*.cov" -type f -print0 | xargs -0 /bin/rm -f

.PHONY: init test