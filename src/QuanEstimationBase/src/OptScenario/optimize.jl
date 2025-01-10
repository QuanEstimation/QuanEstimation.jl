function optimize!(scheme, opt; algorithm=autoGRAPE(), obj=QFIM_obj(), savefile=false, )
    show(scheme) # io1
    output = Output(opt; save=savefile)
    optimize!(opt, algorithm, obj, scheme, output)
    show(obj, output) 
end  

