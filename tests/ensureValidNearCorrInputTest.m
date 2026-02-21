function tests = ensureValidNearCorrInputTest
addpath("../src")
tests = functiontests(localfunctions);
end

% This function should not change an already valid correlation matrix
function testNoChange(testCase)
    dat = [[1 2 3];
           [12 15 14];
           [3 2 5];];
    mat = corr(dat);
    out = ensureValidNearCorrInput(mat,1e-3);
    verifyEqual(testCase,out,mat);
end
