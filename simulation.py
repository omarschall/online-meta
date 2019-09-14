"""simulation.py

Author: @omarschall, 9-13-2019"""


class Simulation:

    def __init__(self, model, train_loader, test_loader, optimizer, Hess_est_r=0.001, mlr=0.000001,
                 **kwargs):

        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.Hess_est_r = Hess_est_r
        self.mlr = mlr

        self.report_interval = 1000

        self.__dict__.update(kwargs)

    def run(self, mode='train'):

        if mode == 'train':
            data_loader = self.train_loader
        elif mode == 'test':
            data_loader = self.test_loader

        for batch_idx, (data, target) in enumerate(data_loader):



    def train_step(self, batch_idx, data, target):

        self.optimizer.zero_grad()
        output = self.model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % report_interval == 0 and batch_idx > 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

        ### --- METALEARNING --- ###

        # Approximate the Hessian
        A = model.unflatten_array(model.A)

        model_plus = copy(model)
        for p, a in zip(model_plus.parameters(), A):
            perturbation = torch.from_numpy(r * a).type(torch.FloatTensor)
            p.data += perturbation

        model_plus.train()
        output = model_plus(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        model_minus = copy(model)
        for p, a in zip(model_minus.parameters(), A):
            p.data -= torch.from_numpy(r * a).type(torch.FloatTensor)

        model_minus.train()
        output = model_plus(data)
        loss = F.nll_loss(output, target)
        loss.backward()

        g_plus = [p.grad.data for p in model_plus.parameters()]
        g_minus = [p.grad.data for p in model_minus.parameters()]
        Q = (model.flatten_array(g_plus) - model.flatten_array(g_minus)) / (2 * r)

        test_grad = get_val_grad(model, test_loader=test_loader)
        model.UORO_update_step(Q)
        new_eta = model.get_updated_eta(mlr, test_grad=test_grad)

        # set_trace()

        for lr, eta in zip(optimizer.param_groups[0]['lr'], new_eta):
            lr.data = torch.from_numpy(eta).type(torch.FloatTensor)

    def train(model, train_loader, test_loader, optimizer, epoch, r=0.001, mlr=0.00001,
              report_interval=1000):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % report_interval == 0 and batch_idx > 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))

            ### --- METALEARNING --- ###

            # Approximate the Hessian
            A = model.unflatten_array(model.A)

            model_plus = copy(model)
            for p, a in zip(model_plus.parameters(), A):
                perturbation = torch.from_numpy(r * a).type(torch.FloatTensor)
                p.data += perturbation

            model_plus.train()
            output = model_plus(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            model_minus = copy(model)
            for p, a in zip(model_minus.parameters(), A):
                p.data -= torch.from_numpy(r * a).type(torch.FloatTensor)

            model_minus.train()
            output = model_plus(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            g_plus = [p.grad.data for p in model_plus.parameters()]
            g_minus = [p.grad.data for p in model_minus.parameters()]
            Q = (model.flatten_array(g_plus) - model.flatten_array(g_minus)) / (2 * r)

            test_grad = get_val_grad(model, test_loader=test_loader)
            model.UORO_update_step(Q)
            new_eta = model.get_updated_eta(mlr, test_grad=test_grad)

            # set_trace()

            for lr, eta in zip(optimizer.param_groups[0]['lr'], new_eta):
                lr.data = torch.from_numpy(eta).type(torch.FloatTensor)

def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def get_val_grad(model, test_loader):
    test_model = copy(model)
    test_model.train()
    for data, target in test_loader:
        # set_trace()
        # TODO: MAKE THIS SAMPLE FROM DIFFERENT PARTS OF THE TEST DATA!
        output = test_model(data)
        test_loss = F.nll_loss(output, target)
        test_loss.backward()
        break

    return [p.grad.data.numpy() for p in test_model.parameters()]