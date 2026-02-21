function tests = randomCorrelationTest
addpath("../src")
tests = functiontests(localfunctions);
end

% Test if it's PSD and unit diagonal
function testIsCorrelation(testCase)
    sz = 5;
    mat = randomCorrelationMatrix(sz);
    verifyGreaterThan(testCase,min(eig(mat)),0);
    verifyEqual(testCase,mat',mat);
    verifyEqual(testCase,diag(mat),ones(sz,1));
end